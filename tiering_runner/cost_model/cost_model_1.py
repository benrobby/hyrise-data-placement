from typing import Dict, List

from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.helpers.globals import SEGMENT_ACCESS_PATTERNS
from tiering_runner.helpers.types import TieringDevice


class CostModel1(CostModel):
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_cost(
        self, segment, calibration_values_per_datatype: Dict[str, List[float]], device: TieringDevice
    ):

        weights_to_use = calibration_values_per_datatype["float"]

        cost = (
            segment[SEGMENT_ACCESS_PATTERNS[0]] * weights_to_use[0]
            + segment[SEGMENT_ACCESS_PATTERNS[1]] * weights_to_use[1]
            + segment[SEGMENT_ACCESS_PATTERNS[2]] * weights_to_use[2]
            + segment[SEGMENT_ACCESS_PATTERNS[3]] * weights_to_use[3]
        )

        return cost
