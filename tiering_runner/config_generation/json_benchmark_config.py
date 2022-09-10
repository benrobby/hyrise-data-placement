from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json
from tiering_runner.helpers.types import TieringDevice


@dataclass_json
@dataclass
class TieringBenchmarkSpec:
    id: int
    sorted_id: int
    benchmark_name: str
    scale_factor: float
    scale_factor_multiplier: float
    optimization_method: str
    num_clients: int
    num_cores: int
    benchmark_queries: List[str]
    benchmark_queries_to_run: List[str]
    devices: List[TieringDevice]
    evaluation_benchmark_runs: int
    evaluation_benchmark_time_s: int
    shuffle_queries: bool
    tiering_config_to_use: str
    calibration_data_to_use: str
    cost_model: str
    dollar_budget_cents: float
    runtime_budget_seconds: float
    runtime_percentage: float
    objective_mode: str
    meta_segments_sf: int
    encoding: str

    @staticmethod
    def csv_header():
        return 'id,sorted_id,benchmark_name,scale_factor,scale_factor_multiplier,optimization_method,cost_model,num_clients,num_cores,"benchmark_queries","benchmark_queries_to_run","devices",evaluation_benchmark_runs,evaluation_benchmark_time_s,dollar_budget_cents,runtime_budget_seconds,runtime_percentage,objective_mode,meta_segments_sf,encoding'

    def m_scale_factor(self):
        return self.scale_factor * self.scale_factor_multiplier

    def m_devices(self):
        return [
            TieringDevice(
                d.device_name,
                d.capacity_GB * self.scale_factor_multiplier * self.meta_segments_sf,
                d.id,
                d.dollar_cents_per_GB,
                [
                    opt * self.scale_factor_multiplier
                    for opt in d.discrete_capacity_option_bytes
                ],
                d.calibration_id,
            )
            for d in self.devices
        ]

    def to_csv_row(self):
        return f'{self.id},{self.sorted_id},{self.benchmark_name},{self.m_scale_factor()},{self.scale_factor_multiplier},{self.optimization_method},{self.cost_model},{self.num_clients},{self.num_cores},"{self.benchmark_queries}","{self.benchmark_queries_to_run}","{self.m_devices()}",{self.evaluation_benchmark_runs},{self.evaluation_benchmark_time_s},{self.dollar_budget_cents},{self.runtime_budget_seconds},{self.runtime_percentage},{self.objective_mode},{self.meta_segments_sf},{self.encoding}'
