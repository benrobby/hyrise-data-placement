import logging
from typing import List

import pandas as pd
from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.determine_tiering.determine_tiering_greedy_utils import sort_devices
from tiering_runner.determine_tiering.determine_tiering_utils import (
    get_device_info,
    get_mappings,
)
from tiering_runner.determine_tiering.tiering_algorithm import TieringAlgorithm
from tiering_runner.helpers.annotations import static_function_vars
from tiering_runner.helpers.globals import (
    OBJECTIVE_MODE_DEVICE_BUDGET,
    SEGMENT_ACCESS_PATTERNS,
    TIERING_CONFIG_DIR,
)
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import (
    DeviceCalibrationResults,
    QueryResultT,
    SegmentTieringAssignment,
    TieringAlgorithmInput,
    TieringAlgorithmMappings,
    TieringAlgorithmResult,
    TieringDevice,
)

APD = [
    "sequential_accesses_runtime",
    "random_accesses_runtime",
    "monotonic_accesses_runtime",
    "single_point_accesses_runtime",
]
ALL_APD = [
    "random_single_chunk_accesses_runtime",
    "random_multiple_chunk_accesses_runtime",
] + APD

logger = logging.getLogger("greedy")


def compute_access_pattern_calibration_weights_diff(device_1, device_2):
    if device_2 is None:
        weights = [float(device_1[f"{p}_mean"]) for p in APD]
        return {
            "float": weights,
            "string": weights,
        }

    c1 = device_1.to_csv().replace("\n", ";")
    c2 = device_2.to_csv().replace("\n", ";")
    access_pattern_calibration_weights_per_datatype = {"float": [], "string": []}

    logger.debug(f"device_1: {c1}, device_2: {c2}")
    for p in APD:
        for d in ["float", "string"]:
            # access_pattern_calibration_weights_per_datatype.append(device_df[p].sum())
            ap_difference = float(device_2[f"{p}_{d}"]) - float(device_1[f"{p}_{d}"])
            assert (
                ap_difference != 0
            ), f"diff should not be 0: {ap_difference} for {p} {d}"
            # can be negative (then the other device wants it more)
            access_pattern_calibration_weights_per_datatype[d].append(ap_difference)

    return access_pattern_calibration_weights_per_datatype


def assign_current_binary_decision_costs_to_segments(
    segments_df: pd.DataFrame, current_decision_device_1, current_decision_device_2
):
    current_decision_device_1_id = int(current_decision_device_1["device_id"])
    current_decision_device_2_id = (
        int(current_decision_device_2["device_id"])
        if current_decision_device_2 is not None
        else None
    )

    # logger.debug(f"segments_df: {segments_df.columns} {segments_df}")

    cost_col_name = f"cost_binary_decision_{current_decision_device_1_id}_{current_decision_device_2_id}"

    if current_decision_device_2 is None:
        segments_df[cost_col_name] = segments_df[
            f"cost_device_{current_decision_device_1_id}"
        ]
    else:
        segments_df[cost_col_name] = (
            segments_df[f"cost_device_{current_decision_device_2_id}"]
            - segments_df[f"cost_device_{current_decision_device_1_id}"]
        )

    segments_df[cost_col_name] = segments_df[cost_col_name].astype(float)
    segments_df["value"] = segments_df[cost_col_name]

    return segments_df


def sort_segments(
    segments_df: pd.DataFrame,
    devices_sorted: pd.DataFrame,
    run_identifier: str,
    cost_model: CostModel,
    current_device_index=0,
) -> pd.DataFrame:

    logger.debug(
        f"Sorting segments with devices in current device {current_device_index}"
    )
    assert 0 <= current_device_index and current_device_index < len(devices_sorted)

    current_decision_device_1 = devices_sorted.iloc[[current_device_index]]
    current_decision_device_2 = (
        devices_sorted.iloc[[current_device_index + 1]]
        if current_device_index < len(devices_sorted) - 1
        else None
    )

    segments_df = assign_current_binary_decision_costs_to_segments(
        segments_df, current_decision_device_1, current_decision_device_2
    )

    segments_sorted = segments_df.sort_values(by=["value"], ascending=False)
    segments_sorted.to_csv(
        TIERING_CONFIG_DIR
        / f"{run_identifier}_{current_device_index}_greedy_segments_sorted.csv"
    )
    return segments_sorted


def sort_devices_and_segments(
    segments_df: QueryResultT,
    mappings: TieringAlgorithmMappings,
    run_identifier: str,
    cost_model: CostModel,
    current_device_index=0,
):

    devices_sorted = sort_devices(segments_df, mappings, run_identifier)
    segments_sorted = sort_segments(
        segments_df, devices_sorted, run_identifier, cost_model, current_device_index
    )

    return devices_sorted, segments_sorted


class DetermineTieringGreedy(TieringAlgorithm):
    def __init__(self, input: TieringAlgorithmInput):
        super().__init__(input)

    def set_up_solver(self):
        return

    @timed
    def _solve(self):
        input = self.input
        mappings = input.mappings
        segments_df = input.meta_segments.copy(deep=True)

        tiering_config = []
        objective_value = 0
        assert (
            input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET
        ), f"Only {OBJECTIVE_MODE_DEVICE_BUDGET} is supported"

        logger.debug(mappings)
        devices_sorted = sort_devices(segments_df, mappings, input.run_identifier)

        # assign segments to devices (start with highest segment value and lowest device score)
        num_segments = len(segments_df)
        logger.debug(f"num_segments: {num_segments}")

        @static_function_vars(devices_index=0)
        def assign_segment_to_next_free_device(segment) -> SegmentTieringAssignment:
            table_id = int(segment["table_id"])
            column_id = int(segment["column_id"])
            chunk_id = int(segment["chunk_id"])
            segment_size_bytes = int(segment["size_in_bytes"])

            previous_device = (
                devices_sorted.iloc[
                    [assign_segment_to_next_free_device.devices_index - 1]
                ]
                if assign_segment_to_next_free_device.devices_index > 0
                else None
            )
            device = devices_sorted.iloc[
                [assign_segment_to_next_free_device.devices_index]
            ]
            device_id, budget_bytes, used_bytes = get_device_info(device, mappings)

            def create_assignment_to_device(d, dev_index: int):
                dev_id, _, used_bytes = get_device_info(d, mappings)

                assignment = SegmentTieringAssignment(
                    table_id,
                    column_id,
                    chunk_id,
                    dev_id,
                    mappings.table_ids_to_names[table_id],
                    mappings.device_ids_to_names[dev_id],
                    segment_size_bytes,
                    sum(int(segment[p]) for p in SEGMENT_ACCESS_PATTERNS),
                )
                devices_sorted.iloc[
                    dev_index,
                    devices_sorted.columns.get_loc("used_bytes"),
                ] = (
                    used_bytes + segment_size_bytes
                )
                objective_for_segment = float(segment["value"]) * (
                    float(d["runtime_performance_mean"]) / 10**9
                )

                return assignment, objective_for_segment, False

            if previous_device is not None:
                _, p_budget_bytes, p_used_bytes = get_device_info(
                    previous_device, mappings
                )
                if p_used_bytes + segment_size_bytes <= p_budget_bytes:
                    # only backtracking one device maximum
                    return create_assignment_to_device(
                        previous_device,
                        assign_segment_to_next_free_device.devices_index - 1,
                    )

            if used_bytes + segment_size_bytes <= budget_bytes:
                return create_assignment_to_device(
                    device, assign_segment_to_next_free_device.devices_index
                )
            else:
                logger.info(
                    f"Device full: {device_id} {used_bytes} {segment_size_bytes} {budget_bytes}"
                )
                if (
                    assign_segment_to_next_free_device.devices_index
                    >= len(devices_sorted) - 1
                ):
                    raise RuntimeError(
                        f"Not enough device capacity for segments: {device_id}, {len(devices_sorted)}"
                    )
                assign_segment_to_next_free_device.devices_index += 1
                assignment, obj, _ = assign_segment_to_next_free_device(segment)
                return assignment, obj, True

        def assign_sorted_segments(remaining_segments):
            nonlocal objective_value
            nonlocal tiering_config

            segment_index = -1
            for _, segment in remaining_segments.iterrows():
                (
                    assignment,
                    objective_for_segment,
                    device_full,
                ) = assign_segment_to_next_free_device(segment)
                if device_full:
                    return segment_index
                tiering_config.append(assignment)
                objective_value += objective_for_segment

                segment_index += 1

            return segment_index

        remaining_segments = segments_df
        while len(remaining_segments) > 0:
            logger.debug(f"remaining_segments: {len(remaining_segments)}")
            remaining_segments = sort_segments(
                remaining_segments,
                devices_sorted,
                input.run_identifier,
                input.cost_model,
                assign_segment_to_next_free_device.devices_index,
            )
            segment_index = assign_sorted_segments(remaining_segments)
            logger.debug(
                f"Finished filling current device, switching to the next. Segment index is {segment_index}, tiering_config: {len(tiering_config)}"
            )
            remaining_segments = remaining_segments.iloc[segment_index + 1 :]

        assert len(tiering_config) == len(
            segments_df
        ), f"{len(tiering_config)} != {len(segments_df)}"
        self.tiering_df = pd.DataFrame(tiering_config)
        self.objective_value = objective_value

    def solve(self):
        res = self._solve()
        self.runtime_ms = res.runtime_milliseconds

    def get_solver_result(self) -> TieringAlgorithmResult:
        return (
            self.tiering_df,
            self.input.mappings.device_ids_to_names,
            self.input.mappings.table_ids_to_names,
            self.objective_value,
            False,
            self.runtime_ms,
        )
