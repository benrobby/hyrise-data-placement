import logging

import pandas as pd
from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.determine_tiering.determine_tiering_utils import set_segment_index
from tiering_runner.helpers.globals import SEGMENT_ACCESS_PATTERNS, TIERING_CONFIG_DIR
from tiering_runner.helpers.types import BenchmarkQueriesT
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface
from tiering_runner.hyrise_server.hyrise_server import HyriseServer
from tiering_runner.run_tiering.get_meta_segments_delta import get_meta_segments_delta


def run_benchmarks_to_fill_access_counters(
    benchmark_queries: BenchmarkQueriesT, server: HyriseServer, args
):
    # run benchmarks to fill access counters
    # todo: how many runs? -> randomize a bit to not have half of the segments with exactly the same access counter
    logging.info("Running a few queries to fill access counters")
    hyrise_client = HyriseInterface(server)
    logging.info(
        f"Queries to run to fill access counters (from args): {args.run_benchmarks_init_queries}"
    )
    for query_name, query_strings in benchmark_queries.items():
        if (
            query_name in args.run_benchmarks_init_queries
            or args.run_benchmarks_init_queries == ["*"]
        ):
            hyrise_client.run_benchmark_query(query_strings[0], query_name)


def init_meta_segments(meta_segments):
    table_names_to_ids = {
        t: i for i, t in enumerate(meta_segments["table_name"].unique())
    }
    meta_segments["table_id"] = meta_segments["table_name"].map(table_names_to_ids)
    set_segment_index(meta_segments)
    return meta_segments


def get_meta_segments(
    args,
    server: HyriseServer,
    spec_id,
    server_was_newly_created,
    benchmark_queries: BenchmarkQueriesT,
):
    logging.info("Getting access counters")

    if args.tiering_meta_segments_file is not None:
        logging.info(
            "Using meta segments from file (this should probably only be used for debugging)"
        )
        meta_segments = pd.read_csv(args.tiering_meta_segments_file)
        return init_meta_segments(meta_segments)

    logging.info("Getting initial meta segments from Hyrise")
    meta_segments_old = HyriseInterface(server).get_meta_segments(should_log=False)
    meta_segments_old = init_meta_segments(meta_segments_old)

    run_benchmarks_to_fill_access_counters(benchmark_queries, server, args)

    logging.info("Getting new meta segments from Hyrise and computing the difference")
    meta_segments_new = HyriseInterface(server).get_meta_segments(should_log=False)
    meta_segments_new = init_meta_segments(meta_segments_new)

    meta_segments_diff = get_meta_segments_delta(meta_segments_new, meta_segments_old)

    for ap in SEGMENT_ACCESS_PATTERNS:
        meta_segments_diff[ap][meta_segments_diff[ap] < 0] = 0

    return meta_segments_diff


def get_meta_segments_raw(
    args,
    server: HyriseServer,
    spec_id,
):
    logging.info("Getting access counters")

    if args.tiering_meta_segments_file is not None:
        logging.info(
            "Using meta segments from file (this should probably only be used for debugging)"
        )
        meta_segments = pd.read_csv(args.tiering_meta_segments_file)
        return init_meta_segments(meta_segments)

    logging.info("Getting meta segments from Hyrise")
    meta_segments = HyriseInterface(server).get_meta_segments(should_log=False)
    meta_segments = init_meta_segments(meta_segments)

    meta_segments.to_csv(TIERING_CONFIG_DIR / f"{spec_id}_meta_segments.csv", sep=",")

    return meta_segments


cached_meta_segments = None


def scale_meta_segments(meta_segments: pd.DataFrame, spec: TieringBenchmarkSpec):
    if spec.meta_segments_sf == 1:
        return meta_segments

    dfs = []
    for i in range(spec.meta_segments_sf):
        df = meta_segments
        table_df = df.groupby("table_name")["chunk_id"].max().reset_index()
        df = pd.merge(df, table_df, on="table_name", suffixes=("", "_table_max"))
        df["chunk_id"] = df["chunk_id"] + i * (df["chunk_id_table_max"] + 1)
        dfs.append(df)

    meta_segments_scaled = pd.concat(dfs, ignore_index=True)

    meta_segments_scaled.to_csv(
        TIERING_CONFIG_DIR / f"{spec.id}_meta_segments_scaled.csv", sep=","
    )
    return meta_segments_scaled


PREVIOUS_BENCHMARK_QUERIES = None


def get_meta_segments_cached(
    args,
    server: HyriseServer,
    spec: TieringBenchmarkSpec,
    server_was_newly_created,
    benchmark_queries: BenchmarkQueriesT,
) -> pd.DataFrame:

    global PREVIOUS_BENCHMARK_QUERIES
    global cached_meta_segments
    if (
        not server_was_newly_created
        and cached_meta_segments is not None
        and PREVIOUS_BENCHMARK_QUERIES == list(benchmark_queries)
    ):
        logging.info("Using cached meta segments")
        meta_segments = cached_meta_segments.copy(deep=True)
        PREVIOUS_BENCHMARK_QUERIES = list(benchmark_queries)
    else:
        logging.info("Getting new meta segments")
        cached_meta_segments = get_meta_segments(
            args, server, spec.id, server_was_newly_created, benchmark_queries
        )
        meta_segments = cached_meta_segments.copy(deep=True)

    meta_segments_scaled = scale_meta_segments(meta_segments, spec)

    return meta_segments_scaled
