import logging
from datetime import datetime
from time import time
from typing import List

from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.helpers.globals import (
    RUN_BENCHMARKS_MODE_NORMAL,
    RUN_BENCHMARKS_MODE_TIMESERIES,
    SPEC_RUNS_FILE,
)
from tiering_runner.helpers.types import (
    BenchmarkConfig,
    DeviceCalibrationResults,
    HyriseServerConfig,
)
from tiering_runner.run_benchmarks.benchmark_query_worker import (
    run_query_worker_threads,
)
from tiering_runner.run_benchmarks.create_server import create_or_reuse_server
from tiering_runner.run_benchmarks.run_benchmarks_helpers import (
    append_runtime_results,
    compare_spec_with_previous,
    get_benchmark_queries,
)
from tiering_runner.run_tiering.get_meta_segments import get_meta_segments_cached
from tiering_runner.run_tiering.run_tiering_async import start_tiering_worker_async
from tiering_runner.run_tiering.run_tiering_sync import run_tiering_sync


def run_benchmarks(
    args,
    server_config: HyriseServerConfig,
    calibration_data: DeviceCalibrationResults,
    benchmark_specs: List[TieringBenchmarkSpec],
):

    if args.run_benchmarks_sort_benchmark_specs:
        benchmark_specs.sort(
            key=lambda x: (
                x.benchmark_name,
                x.m_scale_factor(),
                x.m_devices(),
                x.optimization_method,
                x.num_clients,
                x.num_cores,
            )
        )
        for i, b in enumerate(benchmark_specs):
            b.sorted_id = i

    server = None

    for spec in benchmark_specs:
        logging.info(f"RUNNING BENCHMARK SPEC: {spec}")
        start_time = time()
        benchmark_queries = None

        if args.run_benchmarks_mode == RUN_BENCHMARKS_MODE_NORMAL:
            # to make sure that both optimization methods start with the same access counters
            force_create_new_server = False
        elif args.run_benchmarks_mode == RUN_BENCHMARKS_MODE_TIMESERIES:
            # tiering_worker should restart with a different optimization method and device
            # since it only focuses on dynamic workloads (changing queries)
            force_create_new_server = compare_spec_with_previous(
                benchmark_specs,
                spec.sorted_id,
                lambda x, y: x.optimization_method != y.optimization_method
                or x.devices != y.devices,
            )
        else:
            raise ValueError(f"Unknown run_benchmarks_mode: {args.run_benchmarks_mode}")

        if (
            not args.run_benchmarks_hyrise_evaluation
            and not args.tiering_apply_configuration
            and not args.run_benchmarks_query_generation
            and args.tiering_meta_segments_file is not None
        ):
            server, server_was_newly_created = None, True
        else:
            server, server_was_newly_created = create_or_reuse_server(
                server,
                server_config,
                spec,
                args,
                force_create_new_server=force_create_new_server,
            )

        benchmark_queries = get_benchmark_queries(args, server, spec)
        benchmark_config = BenchmarkConfig(
            server.config,
            spec.optimization_method,
            spec.num_clients,
            benchmark_queries,
            spec.benchmark_queries_to_run,
            loop_executions=(
                float("inf")
                if (
                    spec.num_clients > 1 or spec.evaluation_benchmark_time_s is not None
                )
                else spec.evaluation_benchmark_runs
            ),
            shuffle_queries=spec.shuffle_queries,
            evaluation_benchmark_time_s=spec.evaluation_benchmark_time_s,
            warmup_runs=args.run_benchmarks_warmup_runs,
            vis_queries=args.run_benchmarks_vis_queries,
            spec_id=spec.id,
        )

        if server_was_newly_created:

            if args.run_benchmarks_mode == RUN_BENCHMARKS_MODE_TIMESERIES:
                meta_segments = get_meta_segments_cached(
                    args, server, spec, server_was_newly_created, benchmark_queries
                )
                logging.info(
                    "Timeseries mode, running tiering sync once and then starting tiering worker"
                )
                run_tiering_sync(
                    server,
                    benchmark_config,
                    spec,
                    calibration_data,
                    args,
                    meta_segments,
                )  # apply tiering sync so we have one assignment to start with
                start_tiering_worker_async(
                    server,
                    calibration_data,
                    args,
                    spec.optimization_method,
                    spec.cost_model,
                    spec.m_devices(),
                    benchmark_queries,
                )

        if args.run_benchmarks_mode == RUN_BENCHMARKS_MODE_NORMAL:

            meta_segments = get_meta_segments_cached(
                args, server, spec, server_was_newly_created, benchmark_queries
            )

            run_tiering_sync(
                server,
                benchmark_config,
                spec,
                calibration_data,
                args,
                meta_segments,
            )

        if args.run_benchmarks_hyrise_evaluation:

            runtime_results, measurement_successful = run_query_worker_threads(
                server, benchmark_config
            )
            if not measurement_successful:
                logging.info("Measurement not successful, trying again for this spec")
                for i in range(2):
                    logging.info("Creating new Server")
                    server, server_was_newly_created = create_or_reuse_server(
                        server,
                        server_config,
                        spec,
                        args,
                        force_create_new_server=True,
                    )
                    runtime_results, measurement_successful = run_query_worker_threads(
                        server, benchmark_config
                    )
                    if measurement_successful:
                        break

            if not measurement_successful:
                logging.warn("Could not perform measurement for this spec")
            else:
                append_runtime_results(runtime_results, benchmark_config, spec)

        logging.info(
            f"BENCHMARK SPEC FINISHED IN {time() - start_time} seconds : {spec}"
        )
        with open(SPEC_RUNS_FILE, "a") as f:
            f.write(spec.to_csv_row() + f",{datetime.today().isoformat()}\n")
