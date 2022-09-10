# approach as in Markus Dreseler

import logging
import warnings
from typing import List

import pandas as pd
from ortools.algorithms import pywrapknapsack_solver
from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.determine_tiering.determine_tiering_greedy import (
    assign_current_binary_decision_costs_to_segments,
    compute_access_pattern_calibration_weights_diff,
    get_device_info,
)
from tiering_runner.determine_tiering.determine_tiering_greedy_utils import sort_devices
from tiering_runner.determine_tiering.determine_tiering_utils import set_segment_index
from tiering_runner.determine_tiering.tiering_algorithm import TieringAlgorithm
from tiering_runner.helpers.globals import (
    OBJECTIVE_MODE_DEVICE_BUDGET,
    SEGMENT_ACCESS_PATTERNS,
    TIERING_CONFIG_DIR,
)
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import (
    SegmentTieringAssignment,
    TieringAlgorithmInput,
    TieringAlgorithmMappings,
    TieringAlgorithmResult,
)

logger = logging.getLogger("determine_tiering_knapsack")
pd.options.mode.chained_assignment = None


@timed
def run_solver(solver):
    return solver.Solve()


def solve_knapsack(segments_df, device_0_capacity_bytes: int):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackTieringSolver",
    )
    solver.set_time_limit(60)  # in seconds

    values = list(segments_df["value"].astype(int))
    weights = [list(segments_df["size_in_bytes"].astype(int))]
    capacities = [device_0_capacity_bytes]

    logger.info("Initializing knapsackSolver")

    solver.Init(values, weights, capacities)

    logger.info("KnapsackSolver starting")
    res = run_solver(solver)
    logger.info(f"KnapsackSolver finished in {res.runtime_milliseconds} milliseconds")

    return solver, res.result, res.runtime_milliseconds


def get_segment_info(segment):
    table_id = int(segment["table_id"])
    column_id = int(segment["column_id"])
    chunk_id = int(segment["chunk_id"])
    segment_size_bytes = int(segment["size_in_bytes"])

    return table_id, column_id, chunk_id, segment_size_bytes


def assign_segments_to_current_device(
    tiering_config: List[SegmentTieringAssignment],
    unassigned_segments: pd.DataFrame,
    segments_df: pd.DataFrame,
    current_device_capacity: int,
    current_device_id: int,
    mappings: TieringAlgorithmMappings,
    devices_sorted: pd.DataFrame,
    input: TieringAlgorithmInput,
):

    logger.debug(
        f"Considering {len(unassigned_segments)} unassigned segments out of {len(segments_df)} segments in total"
    )
    solver, objective, solver_runtime_ms = solve_knapsack(
        unassigned_segments, current_device_capacity
    )

    num_assigned_segments = 0

    for i, segment in enumerate(unassigned_segments.itertuples()):

        if solver.BestSolutionContains(i):
            target_device = current_device_id
        else:
            continue

        target_device_id, target_dev_budget, target_dev_used = get_device_info(
            devices_sorted.loc[target_device], mappings
        )

        table_id = int(getattr(segment, "table_id"))
        column_id = int(getattr(segment, "column_id"))
        chunk_id = int(getattr(segment, "chunk_id"))
        segment_size_bytes = int(getattr(segment, "size_in_bytes"))

        assignment = SegmentTieringAssignment(
            table_id,
            column_id,
            chunk_id,
            target_device_id,
            mappings.table_ids_to_names[table_id],
            mappings.device_ids_to_names[target_device_id],
            segment_size_bytes,
            sum(int(getattr(segment, p)) for p in SEGMENT_ACCESS_PATTERNS),
        )
        tiering_config.append(assignment)

        new_used_bytes = target_dev_used + segment_size_bytes
        devices_sorted.iloc[
            target_device_id, devices_sorted.columns.get_loc("used_bytes")
        ] = new_used_bytes

        assert (
            new_used_bytes <= target_dev_budget
        ), "new used bytes is greater than budget"

        segments_df.loc[
            (table_id, column_id, chunk_id), "target_device_id"
        ] = target_device_id
        # logger.debug(
        #     f"Assigning segment {table_id}, {column_id}, {chunk_id} to device {target_device} {current_device_id} {target_device_id}"
        # )
        num_assigned_segments += 1

    logger.debug(
        f"Assigned {num_assigned_segments} out of {len(unassigned_segments)} segments to the current device {current_device_id}. Should be remaining: {len(unassigned_segments) - num_assigned_segments}, actually remaining: {len(segments_df[segments_df['target_device_id'] == -1])}"
    )
    segments_df.to_csv(
        TIERING_CONFIG_DIR
        / f"{input.run_identifier}_{current_device_id}_knapsack_segments_df.csv"
    )

    return objective, solver_runtime_ms


class DetermineTieringKnapsack(TieringAlgorithm):
    def __init__(self, input: TieringAlgorithmInput):
        super().__init__(input)

    def set_up_solver(self):
        self.devices_sorted = sort_devices(
            self.input.meta_segments, self.input.mappings, self.input.run_identifier
        )

    def solve(self):
        input = self.input
        segments_df = input.meta_segments.copy(deep=True)
        set_segment_index(segments_df)

        mappings = input.mappings
        devices_sorted = self.devices_sorted

        assert (
            input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET
        ), f"Only {OBJECTIVE_MODE_DEVICE_BUDGET} is supported"

        if len(mappings.device_ids_to_names.keys()) > 2:
            logger.warn(
                "Using extended algorithm with greedy device ordering, iteratively applying knapsack between two devices at a time until all segments are allocated"
            )

        tiering_config = []
        objective = 0
        solver_runtime_ms = 0

        segments_df["target_device_id"] = -1
        segments_df["value"] = 1.0
        segments_df.sort_index(inplace=True)

        unassigned_segments = segments_df
        dev_id = 0

        logger.debug(f"devices_sorted_columns: {devices_sorted.columns}")
        devices_sorted.set_index("device_id", inplace=True, drop=False)

        # for i in range(len(devices_sorted)):
        #     with pd.option_context(
        #         "display.max_rows", None, "display.max_columns", None
        #     ):
        #         logger.debug(f"Device {i}: {devices_sorted.iloc[[i]]}")

        while len(unassigned_segments) > 0:
            assert dev_id < len(
                devices_sorted
            ), "device id is greater than number of devices"
            unassigned_segments = segments_df[segments_df["target_device_id"] == -1]

            current_device = devices_sorted.iloc[dev_id]
            current_device_id, current_device_capacity, _ = get_device_info(
                current_device, mappings
            )

            logger.debug(
                f'Current device: {current_device_id} {dev_id} {current_device["device_name_float"]}'
            )

            if dev_id == len(devices_sorted) - 1:
                unassigned_segments_size = unassigned_segments["size_in_bytes"].sum()
                assert (
                    unassigned_segments_size <= current_device_capacity
                ), f"unassigned segments are greater than last device capacity: {unassigned_segments_size} > {current_device_capacity} {current_device}"

            if dev_id < len(devices_sorted) - 1:
                next_device = devices_sorted.iloc[[dev_id + 1]]

                unassigned_segments = assign_current_binary_decision_costs_to_segments(
                    unassigned_segments, current_device, next_device
                )
                unassigned_segments = unassigned_segments.sort_values(
                    by=["value"], ascending=False
                )

                unassigned_segments["value"] = unassigned_segments["value"].astype(int)
                unassigned_segments["value"] = (
                    unassigned_segments["value"]
                    - unassigned_segments["value"].min()
                    + 1.0
                )  # can not be negative for knapsack
                unassigned_segments.to_csv(
                    TIERING_CONFIG_DIR
                    / f"{input.run_identifier}_{current_device_id}_knapsack_segments.csv"
                )

            obj, solver_run_ms = assign_segments_to_current_device(
                tiering_config,
                unassigned_segments,
                segments_df,
                current_device_capacity,
                current_device_id,
                mappings,
                devices_sorted,
                input,
            )
            objective += obj
            solver_runtime_ms += solver_run_ms

            dev_id += 1
            unassigned_segments = segments_df[segments_df["target_device_id"] == -1]

        self.tiering_config = tiering_config
        self.objective = objective
        self.solver_runtime_ms = solver_runtime_ms

    def get_solver_result(self) -> TieringAlgorithmResult:
        tiering_df = pd.DataFrame(self.tiering_config)

        return (
            tiering_df,
            self.input.mappings.device_ids_to_names,
            self.input.mappings.table_ids_to_names,
            self.objective,
            True,
            self.solver_runtime_ms,
        )
