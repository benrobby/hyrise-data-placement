import atexit
import cProfile
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from distutils.dir_util import copy_tree
from time import sleep, time
from typing import List

import git

from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.helpers.memory_logging import set_up_file_device_logging
from tiering_runner.helpers.types import HyriseServerConfig
from tiering_runner.parse_args import parse_args
from tiering_runner.run_benchmarks.run_benchmarks import run_benchmarks
from tiering_runner.run_calibration.run_calibration import run_calibration

from .helpers.globals import (
    ALL_LOG_DIRS,
    BENCHMARKS_DIR,
    CALIBRATION_DIR,
    LATEST_OUTPUT_DIR,
    LOGS_DIR,
    O2_CALIBRATION_FILE,
    OUTPUT_DIR,
    PYTHON_SCRIPT_RUN_LOG_FILE,
    RUN_NAME,
    RUN_NAME_FILE,
    RUN_NOTES_FILE,
    RUNTIMES_CSV_FILE,
    SPEC_RUNS_FILE,
    TIERING_DEVICES_META_CSV_FILE,
    TIERING_RUNS_FILE,
    TIERING_RUNS_META_CSV_FILE,
    TMP_DIR,
)

for d in ALL_LOG_DIRS:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=(PYTHON_SCRIPT_RUN_LOG_FILE),
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(name)-28s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-22s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


class MyLogger:
    def __init__(self, l, log_level):
        self.l = l
        self.level = log_level
        self.linebuf = ""

    def flush(self):
        return

    def write(self, s):
        for line in s.rstrip().splitlines():
            self.l.log(self.level, line.rstrip())


stderr_logger = logging.getLogger("STDERR")
sl = MyLogger(stderr_logger, logging.ERROR)
sys.stderr = sl


stdout_logger = logging.getLogger("STDOUT")
sl = MyLogger(stdout_logger, logging.INFO)
sys.stdout = sl


LOGGER = logging.getLogger(__name__)

LOGGER.info(sys.argv)


def init_run_files(args):
    with open(RUNTIMES_CSV_FILE, "w") as f:
        f.write(
            "spec_id,sorted_spec_id,benchmark_name,scale_factor,optimization_method,cost_model,num_clients,num_cores,devices,benchmark_queries,client_id,client_run_id,client_max_runs,result_name,duration_milliseconds,start_time_epoch_seconds\n"
        )

    with open(TIERING_DEVICES_META_CSV_FILE, "w") as f:
        f.write(
            "spec_id,sorted_spec_id,tiering_run_id,benchmark_name,scale_factor,optimization_method,cost_model,device_name,device_id,device_budget_bytes,device_used_bytes,device_used_percentage,objective_value_for_entire_config\n"
        )
    with open(TIERING_RUNS_META_CSV_FILE, "w") as f:
        f.write(
            f"{TieringBenchmarkSpec.csv_header()},solver_runtime,e2e_runtime,solver_peak_memory_bytes,e2e_peak_memory_bytes\n"
        )

    with open(SPEC_RUNS_FILE, "w") as f:
        f.write(f"{TieringBenchmarkSpec.csv_header()},timestamp\n")

    with open(TIERING_RUNS_FILE, "w") as f:
        f.write("optimization_method,run_id,run,timestamp,devices\n")

    with open(RUN_NOTES_FILE, "w") as f:
        f.write("")

    with open(RUN_NAME_FILE, "w") as f:
        f.write(RUN_NAME)

    with open(O2_CALIBRATION_FILE, "a") as f:
        f.write(
            "sf,objective_value_min,objective_value_max,all_query_runtime_milliseconds_min,all_query_runtime_milliseconds_max\n"
        )

    for device_name, device_id in args.all_devices:
        if device_name == "DRAM" or "NUMA_" in device_name:
            continue
        set_up_file_device_logging(device_name)

    if args.benchmark_configurations_file is not None:
        shutil.copy(
            args.benchmark_configurations_file,
            str(OUTPUT_DIR)
            + f"/benchmark_{args.benchmark_configurations_file.stem}.json",
        )
    if args.run_config_file is not None:
        shutil.copy(
            args.run_config_file,
            str(OUTPUT_DIR) + f"/run_{args.run_config_file.stem}.json",
        )
        with open(args.run_config_file) as f:
            d = json.load(f)
            if "default_config_path" in d:
                shutil.copy(d["default_config_path"], OUTPUT_DIR / "run_defaults.json")
    if LATEST_OUTPUT_DIR.exists():
        os.system(
            f"unlink {LATEST_OUTPUT_DIR} && ln -s -f {OUTPUT_DIR} {LATEST_OUTPUT_DIR}"
        )
    else:
        os.system(f"ln -s -f {OUTPUT_DIR} {LATEST_OUTPUT_DIR}")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    LOGGER.info(f"Git SHA: {sha}")
    if repo.is_dirty() and not args.ignore_dirty_repo:
        LOGGER.error("Git repo is dirty, commit your changes.")
        exit(1)


def create_server_config(args):
    server_config = HyriseServerConfig(
        args.hyrise_server_executable_path,
        args.hyrise_dir,
        args.port,
        1,
        None,
        1,
        args.job_data_path,
        args.attach_to_running_server,
        args.running_server_is_initialized,
        None,
    )
    return server_config


def run(args):

    benchmark_configurations: List[TieringBenchmarkSpec] = args.benchmark_configurations

    server_config = create_server_config(args)

    calibration_data = run_calibration(server_config, args)

    if args.run_benchmarks:
        run_benchmarks(args, server_config, calibration_data, benchmark_configurations)

    # copy_tree(str(OUTPUT_DIR), str(LATEST_OUTPUT_DIR))


if __name__ == "__main__":
    start_time = time()

    args = parse_args()
    init_run_files(args)

    if args.profile_cpython:
        logging.warn("Profiling python code with cpython")
        cProfile.run("run(args)", filename="profile.out")
    else:
        run(args)

    logging.info(f"Script finished successfully in {time() - start_time} seconds.")
