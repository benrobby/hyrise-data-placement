import ctypes
import inspect
import logging
import threading
from datetime import datetime
from random import shuffle
from time import sleep, time
from timeit import default_timer as timer
from typing import List, Tuple

from psycopg2 import DatabaseError
from tiering_runner.helpers.globals import MEASUREMENT_ALL_QUERIES
from tiering_runner.helpers.types import (BenchmarkConfig, RuntimeResult,
                                          WorkerRuntimeResultT)
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface
from tiering_runner.hyrise_server.hyrise_plugin_interface import \
    HyrisePluginInterface
from tiering_runner.hyrise_server.hyrise_server import HyriseServer


def repeat_benchmarks_until_finished(
    thread_id: int,
    hyrise_client: HyriseInterface,
    logger,
    queries: List[Tuple[str, List[str]]],
    finish_running_event: threading.Event,
    max_num_runs: int,
    server: HyriseServer,
    warmup_runs=1,
    vis_queries=False,
    optimization_method="*",
    spec_id="",
):

    thread_results = []
    run = 0
    res = None
    warmup_runs = 0 if max_num_runs == 1 else int(warmup_runs)
    logger.info(f"Warmup runs is {warmup_runs}")
    while run < max_num_runs + warmup_runs and not finish_running_event.is_set():

        # todo ich koennte Benchmarks jedes Mal neu random sortieren

        all_queries_start_time_epoch_seconds = time()
        all_queries_start_time = timer()
        for query_index, (item_name, item_instances) in enumerate(queries):

            if finish_running_event.is_set() and query_index < (len(queries) * 0.8):
                # don't abort the whole run if we already have more than 80% of the queries
                return thread_results, run

            item_id = run % len(item_instances)
            evaluation_item = item_instances[item_id]
            try:
                plugin_client = HyrisePluginInterface(server)
                # plugin_client.clear_umap_buffer()  # only one client supported

                logger.debug(f"Queries worker {thread_id} starting query {item_name}.")
                if vis_queries:
                    # meta_segments_before = HyriseInterface(server).get_meta_segments()
                    # meta_segments_before.to_csv(
                    #     TIERING_CONFIG_DIR / f"before_meta_segments.csv", sep=","
                    # )
                    plugin_client.visualize_query(
                        evaluation_item, item_name, f"{spec_id}_{optimization_method}"
                    )
                    # meta_segments_after = HyriseInterface(server).get_meta_segments()
                    # meta_segments_after.to_csv(
                    #     TIERING_CONFIG_DIR / f"after_meta_segments.csv", sep=","
                    # )
                else:
                    res = hyrise_client.run_benchmark_query(evaluation_item, item_name)
            except DatabaseError as e:
                exception_text = str(e).replace("\n", "")
                logger.warn(
                    f"Got Database error: {exception_text}. Not recording this run since it probably returned early. Continuing with next query."
                )
                continue
            logger.debug(f"Queries worker {thread_id} finished query {item_name}.")

            if run >= warmup_runs and res is not None:
                thread_results.append(
                    RuntimeResult(
                        item_name,
                        res.start_time_epoch_seconds,
                        res.runtime_milliseconds,
                        str(datetime.today().strftime("%Y-%m-%dT%H:%M:%S.%f")),
                        run,
                    )
                )

        thread_results.append(
            RuntimeResult(
                MEASUREMENT_ALL_QUERIES,
                all_queries_start_time_epoch_seconds,
                (timer() - all_queries_start_time) * 1000.0,
                str(datetime.today().strftime("%Y-%m-%dT%H:%M:%S.%f")),
                run,
            )
        )

        run += 1

    return thread_results, run


def query_worker_thread(
    thread_id: int,
    server: HyriseServer,
    queries: List[Tuple[str, List[str]]],
    finish_running_event: threading.Event,
    max_num_runs: int,
    runtime_results: List[WorkerRuntimeResultT],
    warmup_runs=1,
    vis_queries=False,
    optimization_method="*",
    spec_id="",
):
    logger = logging.getLogger(f"query_worker_thread_{thread_id}")
    logger.info(f"Queries worker {thread_id} starting.")

    try:
        hyrise_client = HyriseInterface(server)
        thread_results, num_runs = repeat_benchmarks_until_finished(
            thread_id,
            hyrise_client,
            logger,
            queries,
            finish_running_event,
            max_num_runs,
            server,
            warmup_runs,
            vis_queries,
            optimization_method,
            spec_id,
        )
    except Exception as e:
        logger.error(
            f"Query worker {thread_id} got unexpected exception and stopped: {e}"
        )
        raise e

    logger.info(
        f"Queries worker {thread_id} finished. This worker had {num_runs} runs."
    )
    runtime_results[thread_id] = thread_results

    return True



def run_query_worker_threads(
    server: HyriseServer,
    benchmark_config: BenchmarkConfig,
) -> Tuple[List[WorkerRuntimeResultT], bool]:
    logging.info(f"Running queries multi-client with config {benchmark_config}")
    start_time = time()
    finish_running_event = threading.Event()

    runtime_results = [None] * benchmark_config.num_clients
    threads = []

    for thread_id in range(benchmark_config.num_clients):
        # start thread
        # run benchmark and measure time

        is_single_client = benchmark_config.num_clients == 1
        queries_to_run = [
            q
            for q in benchmark_config.benchmark_queries.items()
            if q[0] in benchmark_config.benchmark_queries_to_run
            or benchmark_config.benchmark_queries_to_run == ["*"]
        ]
        if not is_single_client and benchmark_config.shuffle_queries and not benchmark_config.vis_queries:
            shuffle(queries_to_run)

        threads.append(
            threading.Thread(
                target=query_worker_thread,
                args=(
                    thread_id,
                    server,
                    queries_to_run,
                    finish_running_event,
                    benchmark_config.loop_executions,
                    runtime_results,
                    benchmark_config.warmup_runs,
                    benchmark_config.vis_queries,
                    benchmark_config.optimization_method,
                    benchmark_config.spec_id,
                ),
            )
        )
        threads[-1].start()

    if (
        benchmark_config.num_clients > 1
        or benchmark_config.evaluation_benchmark_time_s is not None
    ):
        sleep(benchmark_config.evaluation_benchmark_time_s)
        logging.info("Setting finish_running_event, expecting query workers to finish")
        finish_running_event.set()

    successful = True
    for thread in threads:
        join_timeout = (
            90 * 60
            if benchmark_config.evaluation_benchmark_time_s is not None
            else 240 * 60
        )
        thread.join(join_timeout)
        if thread.is_alive():
            logging.warn("Thread joining timed out, ")
            successful = False
            break

    if not successful:
        for thread in threads:
            if thread.is_alive():
                thread.raiseExc(Exception)

    logging.info(
        f"Benchmark execution for {benchmark_config} finished in {time() - start_time} seconds"
    )
    return runtime_results, successful
