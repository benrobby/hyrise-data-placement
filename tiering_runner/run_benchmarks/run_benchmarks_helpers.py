import json
import logging
import threading
from copy import deepcopy
from datetime import datetime
from random import shuffle
from time import sleep, time
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
import pandas as pd
from psycopg2 import DatabaseError
from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.helpers.types import (
    BenchmarkConfig,
    BenchmarkQueriesT,
    DeviceCalibrationResults,
    HyriseServerConfig,
    RuntimeResult,
    TieringConfigMetadata,
    WorkerRuntimeResultT,
)
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface
from tiering_runner.hyrise_server.hyrise_server import (
    HYRISE_SERVER_PROCESS,
    HyriseServer,
)

from ..helpers.globals import (
    ALL_BENCHMARKS,
    RUNTIMES_CSV_FILE,
    SF_DEPENDENT_QUERIES_BENCHMARKS,
    TIERING_CONFIG_DIR,
    short_name_to_benchmark_t,
    to_long_name,
)


# todo this could be cached
def get_benchmark_queries(
    args,
    server: HyriseServer,
    spec: TieringBenchmarkSpec,
) -> BenchmarkQueriesT:

    if args.run_benchmarks_query_generation:
        logging.info("Using Generated Benchmark Queries from Hyrise")
        hyrise_client = HyriseInterface(server)
        queries = hyrise_client.get_benchmark_queries()
    else:
        logging.info("Using Static Benchmark Queries")
        benchmark_name = short_name_to_benchmark_t(spec.benchmark_name)
        if benchmark_name not in ALL_BENCHMARKS:
            raise Exception(f"Unknown benchmark name: {benchmark_name}")

        sf = (
            f"_{spec.m_scale_factor()}"
            if benchmark_name in SF_DEPENDENT_QUERIES_BENCHMARKS
            else ""
        )
        with open(
            args.static_benchmarks_dir
            / f"{benchmark_name[0].lower()}{sf}_queries.json",
            "r",
        ) as f:
            queries = json.load(f)

    # limit number of variants to 10
    for k, v in queries.items():
        queries[k] = v[:10]

    if spec.benchmark_queries != ["*"]:
        queries = {k: v for k, v in queries.items() if k in spec.benchmark_queries}

    logging.debug(json.dumps(queries))

    return queries


def append_runtime_results(
    runtime_results: List[WorkerRuntimeResultT],
    benchmark_config: BenchmarkConfig,
    spec: TieringBenchmarkSpec,
):
    with open(RUNTIMES_CSV_FILE, "a") as runtimes_csv:
        for thread_id, thread_query_runtimes in enumerate(runtime_results):
            for result in thread_query_runtimes:
                runtimes_csv.write(
                    f'{spec.id},{spec.sorted_id},"{spec.benchmark_name}",{spec.m_scale_factor()},"{spec.optimization_method}","{spec.cost_model}",{spec.num_clients},{spec.num_cores},"{[(d.device_name, d.capacity_GB) for d in spec.m_devices()]}","{spec.benchmark_queries}",{thread_id},{result.thread_run_id},{benchmark_config.loop_executions},"{result.name}",{result.duration_milliseconds},{result.start_time_epoch_seconds}\n'
                )


def compare_spec_with_previous(
    benchmark_specs: List[TieringBenchmarkSpec], current_spec_id, comparison_func
):
    if current_spec_id == 0:
        return False
    return comparison_func(
        benchmark_specs[current_spec_id], benchmark_specs[current_spec_id - 1]
    )
