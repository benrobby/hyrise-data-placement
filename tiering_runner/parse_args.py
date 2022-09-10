import argparse
import json
import logging
import warnings
from cmath import inf
from pathlib import Path
from typing import Dict

from dotmap import DotMap

from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.helpers.types import BenchmarkT, TieringDevice

from .helpers.globals import (
    ALL_BENCHMARKS_SHORT,
    ALL_BENCHMARKS_WITH_DEFAULT_SF,
    DRAM_DEVICE,
    OPTIMIZATION_METHODS,
    RUN_BENCHMARKS_MODES,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter(action="ignore", category=RuntimeWarning)


def add_run_config_args(parser):

    parser.add_argument(
        "--hyrise_server_executable_path",
        "-hsep",
        type=str,
    )
    parser.add_argument(
        "--hyrise_dir",
        type=str,
    )
    parser.add_argument("--port", "-p", type=int)
    parser.add_argument(
        "--job_data_path",
        type=str,
    )
    parser.add_argument(
        "--static_benchmarks_dir",
        type=str,
    )
    parser.add_argument(
        "--ignore_dirty_repo",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--attach_to_running_server",
        action="store_true",
        help="don't start hyriseServer from python, but attach to an already running server",
    )
    parser.add_argument(
        "--running_server_is_initialized",
        action="store_true",
    )
    parser.add_argument("--umap_buf_size_bytes", type=int, default=None)
    parser.add_argument(
        "--profile_cpython",
        action="store_true",
    )

    # ====== CALIBRATION ======
    parser.add_argument(
        "--run_calibration",
        "-run_c",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--calibration_static_dram_speedup_factor", type=float)
    parser.add_argument("--calibration_file", type=str)
    parser.add_argument("--calibration_scale_factor", type=float)
    parser.add_argument("--calibration_benchmark_min_time_seconds", type=float)
    parser.add_argument("--calibration_random_data_size_per_device_mb", type=int)
    parser.add_argument("--calibration_monotonic_access_stride", type=int)
    parser.add_argument("--calibration_num_concurrent_threads", type=int)
    parser.add_argument("--calibration_num_reader_threads", type=int)
    parser.add_argument(
        "--calibration_min_max_objective_latency",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--calibration_datatype_modes", type=str)
    parser.add_argument("--calibration_access_patterns", type=str)

    # ====== TIERING SELECTION ======
    parser.add_argument("--tiering_meta_segments_file", type=str)
    parser.add_argument(
        "--tiering_apply_configuration",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--tiering_postprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--tiering_timeseries_config_update_wait_time_s", type=int)
    parser.add_argument(
        "--tiering_sso_threshold",
        type=int,
        help="The default string capacity of your hyrise compiler, strings with size above this will be allocated outside the object",
    )
    parser.add_argument(
        "--tiering_lp_runtime_budget_measure_query_runtime",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--tiering_lp_runtime_budget_all_query_ms_min",
        type=int,
        help="only used for quick debugging, for actual measurements always measure this!",
    )
    parser.add_argument(
        "--tiering_lp_runtime_budget_all_query_ms_max",
        type=int,
        help="only used for quick debugging, for actual measurements always measure this!",
    )
    parser.add_argument(
        "--tiering_lp_runtime_budget_objective_runtime_exponent", type=float
    )
    parser.add_argument("--tiering_lp_min_max_objective_calibration", type=str)
    parser.add_argument(
        "--tiering_lp_use_discrete_device_capacities",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # ====== BENCHMARKS ======

    parser.add_argument(
        "--run_benchmarks",
        "-run_b",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--run_benchmarks_mode", choices=RUN_BENCHMARKS_MODES, type=str)
    parser.add_argument(
        "--run_benchmarks_hyrise_evaluation",
        "-run_e",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--run_benchmarks_query_generation",
        "-run_q",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use fixed TPC-H queries (taken from Umbra demo) for evaluation",
    )
    parser.add_argument(
        "--run_benchmarks_sort_benchmark_specs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--run_benchmarks_init_queries", default=None, nargs="+")
    parser.add_argument("--run_benchmarks_warmup_runs", default=None)
    parser.add_argument(
        "--run_benchmarks_vis_queries",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--run_benchmarks_always_restart_server",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def postprocess_args(args):
    def to_path(path_str: str, assert_exists=False):
        if path_str is None:
            return None
        p = Path(path_str).expanduser().resolve()
        if assert_exists:
            assert p.exists(), f"path does not exist: {p}"
        return p

    args.hyrise_server_executable_path = to_path(args.hyrise_server_executable_path)
    args.hyrise_dir = to_path(args.hyrise_dir)
    args.job_data_path = to_path(args.job_data_path)
    args.static_benchmarks_dir = to_path(args.static_benchmarks_dir)
    args.benchmark_configurations_file = to_path(args.benchmark_configurations_file)
    args.calibration_file = to_path(args.calibration_file)
    args.run_config_file = to_path(args.run_config_file)
    with open(args.benchmark_configurations_file, "r") as f:
        benchmark_configurations = json.load(f)
        l = []
        for b in benchmark_configurations:
            b["sorted_id"] = b["id"]
            l.append(TieringBenchmarkSpec.from_dict(b))
        args.benchmark_configurations = l

    args.all_devices = {
        (device.device_name, device.id)
        for d in args.benchmark_configurations
        for device in d.devices
    }

    args.all_device_names = [d[0] for d in args.all_devices]


def validate_args(args):
    if not args.run_benchmarks_query_generation:
        assert Path(
            args.static_benchmarks_dir
        ).exists(), f"static benchmarks dir not found {args.static_benchmarks_dir}"

    # assert len(set(args.benchmarks)) == len(args.benchmarks), "Duplicate benchmark"

    # for b in args.benchmarks:
    #     assert (
    #         b[0] in ALL_BENCHMARKS_SHORT
    #     ), f"Benchmark {b} must be one of {ALL_BENCHMARKS_SHORT}"

    # assert not (
    #     args.evaluation_benchmark_time_s is not None
    #     and args.evaluation_benchmark_runs is not None
    # ), "only one of evaluation_benchmark_time_s and evaluation_benchmark_runs can be set"
    # if (
    #     args.evaluation_benchmark_time_s is None
    #     and args.evaluation_benchmark_runs is None
    # ):
    #     if all(c == 1 for c in args.clients):
    #         args.evaluation_benchmark_runs = 1
    #     else:
    #         args.evaluation_benchmark_time_s = 60 * 4

    # for device in args.devices:
    #     if device.device_name == "DRAM":
    #         continue
    #     assert Path(
    #         device.device_name
    #     ).exists(), f"device does not exist: {device.device_path}. Consider mounting the device (see scripts/mount_partitions.sh)."

    # if BENCHMARK_JOB[0] in [b[0] for b in args.benchmarks]:
    #     assert Path(
    #         args.job_data_path
    #     ).exists(), f"job data path does not exist: {args.job_data_path}"

    # todo validate benchmark configs
    print(args)


def selective_merge(base_obj: Dict, delta_obj: Dict):
    if not isinstance(base_obj, dict):
        return delta_obj
    common_keys = set(base_obj.keys()).intersection(delta_obj.keys())
    new_keys = set(delta_obj.keys()).difference(common_keys)
    for k in common_keys:
        base_obj[k] = selective_merge(base_obj[k], delta_obj[k])
    for k in new_keys:
        base_obj[k] = delta_obj[k]
    return base_obj


def get_merged_file_config(args_file):
    if "default_config_path" not in args_file:
        return args_file

    with open(args_file["default_config_path"]) as f:
        file_default_conf = json.load(f)
        logger.debug(f"file_default_conf: {file_default_conf}")
        logger.debug(f"args_file: {args_file}")
        merged_default_conf = selective_merge(file_default_conf, args_file)
        return merged_default_conf


def get_merged_config(cmd_args):

    if cmd_args.run_config_file is None:
        return cmd_args

    logger.debug(f"Using parameters from: {cmd_args.run_config_file}")
    with open(cmd_args.run_config_file) as f:
        args_file = json.load(f)

        file_conf = get_merged_file_config(args_file)
        logger.debug(f"file_conf: {file_conf}")

        cmd_dict = {}
        for k, v in cmd_args.__dict__.items():
            if v is not None:
                cmd_dict[k] = v

        cmd_and_file_conf = selective_merge(file_conf, cmd_dict)
        logger.debug(f"cmd_and_file_conf: {cmd_and_file_conf}")

        return DotMap(cmd_and_file_conf)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyrise with tiering configuration"
    )
    # the two main configurations: the benchmarks to run, and meta arguments for the scripts
    parser.add_argument(
        "--benchmark_configurations_file",
        "-bf",
        type=str,
        help="specify this or benchmark-variable args. If specified, overrides all benchmark-variable args",
    )
    parser.add_argument(
        "--run_config_file",
        "-rf",
        type=str,
        help="specify this or the static args",
    )

    # can specify this additionally, will overwrite the values from the configs
    add_run_config_args(parser)

    cmd_args = parser.parse_args()
    logger.debug(f"Cmd args are: {cmd_args}")

    merged_config = get_merged_config(cmd_args)

    postprocess_args(merged_config)
    validate_args(merged_config)

    return merged_config
