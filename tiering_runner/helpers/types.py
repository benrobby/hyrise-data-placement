from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dataclasses_json import dataclass_json
from tiering_runner.cost_model.cost_model import CostModel

BenchmarkT = Tuple[str, str]  # short name, long name
BenchmarksT = List[BenchmarkT]
ColNamesT = Dict[str, int]
BenchmarkQueriesT = Dict[str, List[str]]
QueryResultT = pd.DataFrame


@dataclass_json
@dataclass
class RuntimeResult:
    name: str
    start_time_epoch_seconds: float
    duration_milliseconds: float
    date_string: str
    thread_run_id: int


@dataclass
class DeviceCalibrationResult:
    sequential_accesses_runtime: float
    random_single_chunk_accesses_runtime: float
    random_multiple_chunk_accesses_runtime: float
    random_accesses_runtime: float
    monotonic_accesses_runtime: float
    single_point_accesses_runtime: float
    device_id: int
    device_name: str
    datatype: str


@dataclass
class SegmentTieringAssignment:
    table_id: int
    column_id: int
    chunk_id: int
    device_id: int
    table_name: str
    device_name: str
    segment_size_bytes: int
    access_count_sum: int

    def get_indices(self):
        return self.table_id, self.column_id, self.chunk_id


@dataclass_json
@dataclass(order=True)
class TieringDevice:
    device_name: str
    capacity_GB: float
    id: int
    dollar_cents_per_GB: float = 0.0
    discrete_capacity_option_bytes: List[int] = field(default_factory=list)
    calibration_id: int = None

    def capacity_bytes(self) -> int:
        return int(self.capacity_GB * 1024 * 1024 * 1024)

    def used_bytes_percentage(self, used_bytes: int) -> float:
        capacity = self.capacity_bytes()
        if capacity <= 0:
            return 1.0
        percentage = used_bytes / float(capacity)
        return percentage

    def get_calibration_id(self):
        if self.calibration_id is None:
            return self.id
        return self.calibration_id


DeviceCalibrationResults = Dict[str, DeviceCalibrationResult]

WorkerRuntimeResultT = List[RuntimeResult]

TieringAlgorithmResult = Tuple[
    pd.DataFrame, Dict[int, str], Dict[int, str], float, bool, float
]


@dataclass
class TieringConfigMetadata:
    benchmark_name: str
    scale_factor: float
    optimization_method: str
    devices: List[TieringDevice]


@dataclass
class HyriseServerConfig:
    hyrise_server_executable_path: Path
    hyrise_dir: Path
    port: int
    scale_factor: float
    benchmark_name: BenchmarkT
    cores: int
    job_data_path: Path
    attach_to_running_server: bool
    running_server_is_initialized: bool
    encoding: str


@dataclass
class BenchmarkConfig:
    server_config: HyriseServerConfig
    optimization_method: str
    num_clients: int
    benchmark_queries: BenchmarkQueriesT
    benchmark_queries_to_run: List[str]
    loop_executions: float
    shuffle_queries: bool
    evaluation_benchmark_time_s: int
    warmup_runs: int
    vis_queries: bool
    spec_id: str

    def __str__(self) -> str:
        return """
        benchmark_name={}
        scale_factor={}
        cores={}
        num_clients={}
        optimization_method={}
        loop_executions={}
        len_benchmark_queries={}
        warmup_runs={}
        vis_queries={}
        spec_id={}
        """.format(
            self.server_config.benchmark_name[0],
            self.server_config.scale_factor,
            self.server_config.cores,
            self.num_clients,
            self.optimization_method,
            self.loop_executions,
            len(self.benchmark_queries),
            self.warmup_runs,
            self.vis_queries,
            self.spec_id,
        )


@dataclass
class TieringAlgorithmMappings:
    device_names_to_ids: Dict[str, int]
    device_ids_to_names: Dict[int, str]
    device_ids_to_storage_budgets_bytes: Dict[int, int]
    device_ids_to_calibration_results: DeviceCalibrationResults
    device_ids_to_dollar_cents_per_GB: Dict[int, float]
    device_ids_to_discrete_capacity_option_bytes: Dict[int, List[int]]
    table_names_to_ids: Dict[str, int]
    table_ids_to_names: Dict[int, str]
    device_calibration_ids_to_ids: Dict[int, int]


@dataclass
class DetermineTieringRunMetadata:
    benchmark_name: str
    scale_factor: str
    run_id: str
    run_sorted_id: str
    run_identifier: str


@dataclass
class TieringAlgorithmInput:
    meta_segments: pd.DataFrame
    mappings: TieringAlgorithmMappings
    run_identifier: str
    cost_model: CostModel
    objective_mode: str
    dollar_budget_cents: float
    runtime_budget_percentage: float
    server: Any
    benchmark_config: BenchmarkConfig
    args: Any
    spec: Any
