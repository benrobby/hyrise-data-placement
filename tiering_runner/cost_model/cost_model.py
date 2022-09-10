from typing import Dict, List


class CostModel:
    def __init__(self, args):
        self.args = args

    def get_cost(
        self, segment, calibration_values_per_datatype: Dict[str, List[float]], device
    ):
        raise NotImplementedError()
