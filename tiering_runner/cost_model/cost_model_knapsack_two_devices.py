from typing import Dict, List

from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.helpers.types import TieringDevice

APS = ["sequential_accesses", "random_accesses", "monotonic_accesses", "point_accesses"]

class CostModelKnapsackTwoDevices(CostModel):
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_cost(self, segment, calibration_values_per_datatype: Dict[str, List[float]], device: TieringDevice):
        if device.device_name == "DRAM":
            return 0.0

        value = (segment["sequential_accesses"]
            + segment["random_accesses"] * 100
            + segment["monotonic_accesses"]
            + segment["point_accesses"])

        return value
