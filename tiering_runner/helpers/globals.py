import platform
import sys
from datetime import datetime
from pathlib import Path

BENCHMARK_TPCH = ("TPCH", "TPC-H")
BENCHMARK_TPCDS = ("TPCDS", "TPC-DS")
BENCHMARK_JOB = ("JOB", "Join Order Benchmark")
BENCHMARK_JCCH = ("JCCH", "JCC-H")
ALL_BENCHMARKS = [
    BENCHMARK_TPCH,
    BENCHMARK_TPCDS,
    BENCHMARK_JOB,
    BENCHMARK_JCCH,
]
ALL_BENCHMARKS_SHORT = [b[0] for b in ALL_BENCHMARKS]
ALL_BENCHMARKS_LONG = [b[1] for b in ALL_BENCHMARKS]
ALL_BENCHMARKS_WITH_DEFAULT_SF = [
    (BENCHMARK_TPCH[0], 1),
    (BENCHMARK_TPCDS[0], 1),
    (BENCHMARK_JCCH[0], 1),
    (BENCHMARK_JOB[0], 1),
]
SF_DEPENDENT_QUERIES_BENCHMARKS = [BENCHMARK_JCCH]

DRAM_DEVICE = "DRAM"

OPTIMIZATION_METHOD_LP = "lp"
OPTIMIZATION_METHOD_GREEDY = "greedy"
OPTIMIZATION_METHOD_DRAM = "DRAM"
OPTIMIZATION_METHOD_COLUMN_LP = "column_lp"
OPTIMIZATION_METHOD_KNAPSACK = "knapsack"
OPTIMIZATION_METHODS = [
    OPTIMIZATION_METHOD_LP,
    OPTIMIZATION_METHOD_GREEDY,
    OPTIMIZATION_METHOD_KNAPSACK,
    OPTIMIZATION_METHOD_COLUMN_LP,
]
OPTIMIZATION_METHODS_DOLLAR_BUDGET = [
    OPTIMIZATION_METHOD_LP,
]
OPTIMIZATION_METHODS_RUNTIME_BUDGET = [
    OPTIMIZATION_METHOD_LP,
]

OBJECTIVE_MODE_DEVICE_BUDGET = "DEVICE_BUDGET"
OBJECTIVE_MODE_DOLLAR_BUDGET = "DOLLAR_BUDGET"
OBJECTIVE_MODE_RUNTIME_BUDGET = "RUNTIME_BUDGET"

RUN_BENCHMARKS_MODE_NORMAL = "normal"
RUN_BENCHMARKS_MODE_TIMESERIES = "timeseries"
RUN_BENCHMARKS_MODES = [RUN_BENCHMARKS_MODE_NORMAL, RUN_BENCHMARKS_MODE_TIMESERIES]


def short_name_to_benchmark_t(short_name):
    return next(b for b in ALL_BENCHMARKS if b[0] == short_name)


def to_long_name(short_benchmark_name: str) -> str:
    return next(b[1] for b in ALL_BENCHMARKS if b[0] == short_benchmark_name)


def to_short_name(long_benchmark_name: str) -> str:
    return next(b[0] for b in ALL_BENCHMARKS if b[1] == long_benchmark_name)


def device_name_to_hyrise_device_name(device_name):
    return device_name


def hyrise_device_name_to_device_name(hyrise_device_name):
    return hyrise_device_name


PLUGIN_FILETYPE = ".so" if sys.platform == "linux" else ".dylib"
SCRIPT_EXEC_TIMESTAMP = datetime.today().strftime("%Y-%m-%dT%H%M%S")

RUN_NAME = f"{SCRIPT_EXEC_TIMESTAMP}_{platform.node()}"
OUTPUT_DIR = Path("output").expanduser().resolve() / RUN_NAME
LATEST_OUTPUT_DIR = Path("output_latest").expanduser()  # don't resolve symlink!!
LOGS_DIR = OUTPUT_DIR / "logs"
TMP_DIR = OUTPUT_DIR / "tmp"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"
SYSTEM_UTILIZATION_DIR = OUTPUT_DIR / "system_utilization"
TIERING_CONFIG_DIR = OUTPUT_DIR / "tiering_configs"
TIERING_DIR = OUTPUT_DIR / "tiering_meta"
ALL_LOG_DIRS = [
    LOGS_DIR,
    CALIBRATION_DIR,
    TMP_DIR,
    BENCHMARKS_DIR,
    SYSTEM_UTILIZATION_DIR,
    TIERING_CONFIG_DIR,
    TIERING_DIR,
]

HYRISE_SERVER_RUN_LOG_FILE = LOGS_DIR / "hyrise_server.log"
PYTHON_SCRIPT_RUN_LOG_FILE = LOGS_DIR / "python_script.log"

TIERING_DEVICES_META_CSV_FILE = TIERING_DIR / "tiering_meta.csv"
TIERING_RUNS_META_CSV_FILE = TIERING_DIR / "tiering_runs_meta.csv"
RUNTIMES_CSV_FILE = BENCHMARKS_DIR / "runtimes.csv"
CALIBRATION_FILE = CALIBRATION_DIR / "calibration_results.csv"
SPEC_RUNS_FILE = BENCHMARKS_DIR / "spec_runs.csv"
TIERING_RUNS_FILE = TIERING_DIR / "tiering_runs.csv"
RUN_NOTES_FILE = OUTPUT_DIR / "run_notes.md"
RUN_NAME_FILE = OUTPUT_DIR / "run_name.txt"

O2_CALIBRATION_FILE = CALIBRATION_DIR / "o2_calibration_results.csv"

COST_MODEL_7 = "COST_MODEL_7"
COST_MODEL_6 = "COST_MODEL_6"
COST_MODEL_4 = "COST_MODEL_4"
COST_MODEL_3 = "COST_MODEL_3"
COST_MODEL_2 = "COST_MODEL_2"
COST_MODEL_1 = "COST_MODEL_1"
COST_MODEL_KNAPSACK_TWO_DEVICES = "COST_MODEL_KNAPSACK_TWO_DEVICES"

SEGMENT_ACCESS_PATTERNS = [
    "sequential_accesses",
    "random_accesses",
    "monotonic_accesses",
    "point_accesses",
    "dictionary_accesses",
]

MEASUREMENT_ALL_QUERIES = "All queries"
