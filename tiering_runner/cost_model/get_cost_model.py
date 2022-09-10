from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.cost_model.cost_model_1 import CostModel1
from tiering_runner.cost_model.cost_model_2 import CostModel2
from tiering_runner.cost_model.cost_model_3 import CostModel3
from tiering_runner.cost_model.cost_model_4 import CostModel4
from tiering_runner.cost_model.cost_model_6 import CostModel6
from tiering_runner.cost_model.cost_model_7 import CostModel7
from tiering_runner.cost_model.cost_model_knapsack_two_devices import (
    CostModelKnapsackTwoDevices,
)
from tiering_runner.helpers.globals import (
    COST_MODEL_1,
    COST_MODEL_2,
    COST_MODEL_3,
    COST_MODEL_4,
    COST_MODEL_6,
    COST_MODEL_7,
    COST_MODEL_KNAPSACK_TWO_DEVICES,
)

cost_model_names_to_classes = {
    COST_MODEL_1: CostModel1,
    COST_MODEL_2: CostModel2,
    COST_MODEL_3: CostModel3,
    COST_MODEL_4: CostModel4,
    COST_MODEL_6: CostModel6,
    COST_MODEL_7: CostModel7,
    COST_MODEL_KNAPSACK_TWO_DEVICES: CostModelKnapsackTwoDevices,
}


def get_cost_model(args, cost_model_name) -> CostModel:

    assert (
        cost_model_name in cost_model_names_to_classes.keys()
    ), f"cost model {cost_model_name} not implemented"

    return cost_model_names_to_classes[cost_model_name](args)
