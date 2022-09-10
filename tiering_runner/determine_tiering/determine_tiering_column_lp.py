import logging
import os
import shutil
from typing import Dict

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from tiering_runner.cost_model.cost_model import CostModel
from tiering_runner.determine_tiering.determine_tiering_lp import (
    DetermineTieringLP,
    get_lp_model_params,
    get_tiering_config,
    print_overview,
    write_segment_cost_file_header,
)
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

logger = logging.getLogger("column_lp")


def solve_lp(
    model,
    mappings: TieringAlgorithmMappings,
    lp_segments: pd.DataFrame,
    run_identifier: str,
):
    solver = SolverFactory(
        "gurobi",
    )
    solver.options["threads"] = 8
    solver.options["timeLimit"] = 500  # seconds
    solver.options["MIPGap"] = 0.01  # percent
    logger.info("Solving LP model...")
    result = solver.solve(model, tee=True)

    return result


def add_column_granularity_constraints(model, segment_sizes: pd.DataFrame):
    segment_sizes_with_col_index = segment_sizes.set_index(
        pd.MultiIndex.from_frame(
            segment_sizes[["table_id", "column_id"]],
            names=["table_id", "column_id"],
        ),
    )

    model.Y = pyo.Var(model.T, model.M, model.D, within=pyo.Binary)

    def column_selected_rule(model, t, m):
        return sum(model.Y[t, m, device_id] for device_id in model.D) == int(
            (t, m) in segment_sizes_with_col_index.index
        )  # each col must be on exactly one device and non-existing cols must not be selected

    model.ColumnSelectedConstraint = pyo.Constraint(
        model.T, model.M, rule=column_selected_rule
    )

    def all_column_segments_on_same_device_rule(model, t, m, d):
        num_valid_segments_for_col = sum(
            1 if (t, m, n) in segment_sizes.index else 0 for n in model.N
        )
        return sum(model.X[t, m, n, d] for n in model.N) == (
            num_valid_segments_for_col * model.Y[t, m, d]
        )
        # either select all segments (all X == 1 for n in model.N and Y == 1) or select none (all X == 0 and Y == 0)

    model.AllColumnSegmentsOnSameDeviceConstraint = pyo.Constraint(
        model.T, model.M, model.D, rule=all_column_segments_on_same_device_rule
    )

    # todo ensure that postprocessing doesn't sneak in segments that don't fit (maybe just skip it?)


ID_VARS = ["table_id", "column_id"]


def create_lp_model(
    num_tables: int,
    num_columns: int,
    num_chunks: int,
    device_id_to_storage_budget_MB: Dict[int, int],
    segment_sizes: pd.DataFrame,
    input: TieringAlgorithmInput,
    runtime_objective_budget: float,
    add_segment_selected_constraint=True,
):
    segment_sizes = segment_sizes.set_index(ID_VARS)

    mappings = input.mappings
    logger.debug(
        f"Creating LP model for {num_tables} tables, {num_columns} columns, {num_chunks} chunks."
    )

    model = pyo.ConcreteModel()

    # set of tables
    model.T = pyo.Set(initialize=range(0, num_tables))

    # set of columns
    model.M = pyo.Set(initialize=range(0, num_columns))

    # set of storage devices
    model.D = pyo.Set(initialize=range(0, len(device_id_to_storage_budget_MB)))

    # device dollars (cents per GB)
    model.DD = pyo.Param(
        model.D,
        within=pyo.NonNegativeReals,
        initialize=mappings.device_ids_to_dollar_cents_per_GB,
        mutable=True,
    )

    # device storage budget
    model.DB = pyo.Param(
        model.D,
        within=pyo.NonNegativeReals,
        initialize=device_id_to_storage_budget_MB,
        mutable=True,
    )

    # decision variable to describe the selected configuration option
    model.X = pyo.Var(model.T, model.M, model.D, within=pyo.Binary)

    # todo move all of this up
    for d in model.D:
        segment_sizes[f"cost_device_{d}"] /= 10**6

    segments_df = segment_sizes.reset_index()[
        ID_VARS + ["size_in_bytes"] + [f"cost_device_{d}" for d in model.D]
    ]
    segments_df = segments_df.groupby(ID_VARS).sum().reset_index()

    cost_cols = [c for c in segments_df if c.startswith("cost_device_")]
    segments_cost_df = segments_df.melt(
        id_vars=ID_VARS,
        value_vars=cost_cols,
        value_name="cost",
    )
    segments_cost_df["device_id"] = (
        segments_cost_df["variable"].str.replace("cost_device_", "").astype(int)
    )
    cost_dict = segments_cost_df.set_index(ID_VARS + ["device_id"])["cost"].to_dict()

    logger.debug(f"Cost dict: {cost_dict}")

    model.C = pyo.Param(
        model.T,
        model.M,
        model.D,
        within=pyo.NonNegativeReals,
        initialize=cost_dict,
        mutable=False,
        default=0.0,
    )

    # segment memory consumption
    # def memory_init(model, t, m, n):
    #     try:
    #         return segment_sizes.loc[t, m, n]["size_in_bytes"]
    #     except KeyError:
    #         # logger.debug(f"Segment not found: ({t},{m},{n})")
    #         return 0

    memory_dict = (
        segments_df.groupby(ID_VARS)
        .sum()
        .reset_index()
        .set_index(ID_VARS)["size_in_bytes"]
        .to_dict()
    )
    logger.debug(f"Memory dict: {memory_dict}")

    model.SMC = pyo.Param(
        model.T,
        model.M,
        within=pyo.NonNegativeIntegers,
        initialize=memory_dict,
        default=0,
        mutable=False,
    )

    def runtime(model):
        return sum(
            model.X[table_id, column_id, device_id]
            * model.C[table_id, column_id, device_id]
            for table_id in model.T
            for column_id in model.M
            for device_id in model.D
        )

    def device_memory_usage_byte(model, d):
        return sum(
            (model.X[table_id, column_id, d] * model.SMC[table_id, column_id])
            for table_id in model.T
            for column_id in model.M
        )

    def dollar_cents_cost(model):
        costs = 0
        for d in model.D:
            continuous_size_bytes = device_memory_usage_byte(model, d)

            costs += model.DD[d] * continuous_size_bytes  # discrete_size_bytes

        return costs / 1024**3

    def memory_budget_rule(model, d):  # only per storage device d
        return (device_memory_usage_byte(model, d) / (1024**2)) <= model.DB[d]

    def dollar_budget_rule(model):
        return dollar_cents_cost(model) <= input.dollar_budget_cents

    def runtime_objective_budget_rule(model):
        return runtime(model) <= runtime_objective_budget * 1.001

    if input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET:
        model.Obj = pyo.Objective(rule=runtime)
        model.MemoryBudgetConstraint = pyo.Constraint(model.D, rule=memory_budget_rule)
    else:
        raise Exception(f"Unknown objective mode: {input.objective_mode}")

    if add_segment_selected_constraint:
        # existing segments must be selected, non-existing segments must not be selected
        def segment_selected_rule(model, t, m):
            return sum(model.X[t, m, device_id] for device_id in model.D) == int(
                (t, m) in segment_sizes.index
            )

        model.SegmentSelectedConstraint = pyo.Constraint(
            model.T, model.M, rule=segment_selected_rule
        )

    return model


def get_tiering_config(
    model,
    table_ids_to_names: Dict[int, str],
    device_ids_to_names: Dict[int, str],
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    tiering_config = []

    # segments_df.to_csv("test.csv")
    set_segment_index(segments_df)
    # logger.debug(f"Segments: {segments_df}")

    for item in model.X:
        if round(model.X[item].value) == 1.0:
            try:
                # logger.info(f"Selected segment: {item}")
                (table_id, column_id, device_id) = item
                chunks_df = segments_df.loc[table_id, column_id, :]

                for segment in chunks_df.itertuples():
                    # logger.debug(f"Segment: {segment}")
                    chunk_id = int(getattr(segment, "chunk_id"))
                    size_in_bytes = int(getattr(segment, "size_in_bytes"))
                    tiering_config.append(
                        SegmentTieringAssignment(
                            table_id,
                            column_id,
                            chunk_id,
                            device_id,
                            table_ids_to_names[table_id],
                            device_ids_to_names[device_id],
                            size_in_bytes,
                            sum(
                                int(getattr(segment, p))
                                for p in SEGMENT_ACCESS_PATTERNS
                            ),
                        )
                    )
            except KeyError:
                logger.warning(f"KeyError: {item}")
                # column LP chooses non-existing segments
                continue
    df = pd.DataFrame(tiering_config)
    return df


class DetermineTieringMosaicLP(TieringAlgorithm):
    def __init__(self, input: TieringAlgorithmInput):
        super().__init__(input)
        assert (
            input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET
        ), f"Only {OBJECTIVE_MODE_DEVICE_BUDGET} is currently supported (todo)"

    def set_up_solver(self):
        input = self.input
        # extends the normal lp solution so that the LP only works on column granularity instead of segment granularity
        # this does NOT use the mosaic cost model (table scans, device bandwidth + latency)
        assert (
            input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET
        ), f"Only {OBJECTIVE_MODE_DEVICE_BUDGET} is currently supported (todo)"

        write_segment_cost_file_header(input.run_identifier)

        (
            lp_segments,
            num_tables,
            num_columns,
            num_chunks,
            device_budgets_in_MB,
        ) = get_lp_model_params(input)

        self.model = create_lp_model(
            num_tables,
            num_columns,
            num_chunks,
            device_budgets_in_MB,
            lp_segments,
            input,
            1.0,
            add_segment_selected_constraint=True,
        )
        self.lp_segments = lp_segments

        # add_column_granularity_constraints(self.model, lp_segments)

    @timed
    def _solve(self):
        input = self.input
        mappings = input.mappings

        self.result = solve_lp(
            self.model, mappings, self.lp_segments, input.run_identifier
        )

    def solve(self):
        res = self._solve()
        self.runtime_ms = res.runtime_milliseconds

    def get_solver_result(self) -> TieringAlgorithmResult:
        input = self.input
        mappings = input.mappings
        print_overview(self.result, self.model, mappings.device_ids_to_names)
        self.tiering_config = get_tiering_config(
            self.model,
            mappings.table_ids_to_names,
            mappings.device_ids_to_names,
            self.lp_segments,
        )
        # if file exists move it
        if os.path.exists("./recording000.grbr"):
            shutil.move(
                "./recording000.grbr",
                TIERING_CONFIG_DIR / f"{self.input.run_identifier}_recording.grbr",
            )
        return (
            self.tiering_config,
            mappings.device_ids_to_names,
            mappings.table_ids_to_names,
            self.model.Obj.expr(),
            True,
            self.runtime_ms,
        )
