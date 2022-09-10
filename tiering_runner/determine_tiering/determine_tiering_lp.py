import logging
import os
import shutil
import sys
import traceback
from copy import copy, deepcopy
from math import log, log10, sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from tiering_runner.determine_tiering.determine_tiering_greedy_utils import sort_devices
from tiering_runner.determine_tiering.determine_tiering_lp_discrete import (
    add_discrete_dollar_cents_costs,
)
from tiering_runner.determine_tiering.determine_tiering_utils import (
    bytes_to_gb,
    clamp,
    execute_tiering_algorithm,
    lerp,
    postprocess_tiering_conf,
    set_segment_index,
)
from tiering_runner.determine_tiering.tiering_algorithm import TieringAlgorithm
from tiering_runner.helpers.globals import (
    MEASUREMENT_ALL_QUERIES,
    O2_CALIBRATION_FILE,
    OBJECTIVE_MODE_DEVICE_BUDGET,
    OBJECTIVE_MODE_DOLLAR_BUDGET,
    OBJECTIVE_MODE_RUNTIME_BUDGET,
    SEGMENT_ACCESS_PATTERNS,
    TIERING_CONFIG_DIR,
)
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import (
    SegmentTieringAssignment,
    TieringAlgorithmInput,
    TieringAlgorithmResult,
)
from tiering_runner.hyrise_server.hyrise_plugin_interface import HyrisePluginInterface
from tiering_runner.run_benchmarks.benchmark_query_worker import (
    run_query_worker_threads,
)

logger = logging.getLogger("determine_tiering_lp")
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

ALL_QUERY_RUNTIME_MIN = None
ALL_QUERY_RUNTIME_MAX = None
ALL_QUERY_OBJECTIVE_MIN = None
ALL_QUERY_OBJECTIVE_MAX = None


def get_objective(
    original_input: TieringAlgorithmInput,
    input: TieringAlgorithmInput,
    target_device: pd.DataFrame,
):
    input.mappings.device_ids_to_storage_budgets_bytes = {
        k: 0 for k in input.mappings.device_ids_to_storage_budgets_bytes.keys()
    }
    input.mappings.device_ids_to_storage_budgets_bytes[
        int(target_device.device_id)
    ] = sum(original_input.mappings.device_ids_to_storage_budgets_bytes.values())

    res = execute_tiering_algorithm(DetermineTieringLP, input)
    (tiering_config, _, _, objective_value, _, _) = res.result
    tiering_config = postprocess_tiering_conf(tiering_config, input)
    return tiering_config, objective_value


def get_objective_and_runtime_for_device_assignment(
    original_input: TieringAlgorithmInput,
    input: TieringAlgorithmInput,
    target_device: pd.DataFrame,
    default_runtime: float,
    cached_runtime: float,
    cached_objective: float,
    use_cached: bool,
):

    if use_cached:
        logger.info(
            f"Using cached query runtime for runtime objective budget {cached_runtime}"
        )
        return cached_objective, cached_runtime

    tiering_config, objective_value = get_objective(
        original_input, input, target_device
    )

    if not input.args.tiering_lp_runtime_budget_measure_query_runtime:
        logger.info(
            "Not measuring query runtime for runtime objective budget, make sure that the supplied min and max values are correct for this SF."
        )
        return objective_value, default_runtime

    plugin_client = HyrisePluginInterface(input.server)
    plugin_client.apply_tiering_configuration(tiering_config)
    benchmark_config = deepcopy(input.benchmark_config)
    benchmark_config.num_clients = 1
    benchmark_config.evaluation_benchmark_time_s = None
    benchmark_config.loop_executions = 1
    runtime_results = run_query_worker_threads(input.server, benchmark_config)
    # logger.debug(f"Runtime results: {runtime_results}")
    worker_0_results = runtime_results[0][0]
    all_query_runtimes = [
        r for r in worker_0_results if r.name == MEASUREMENT_ALL_QUERIES
    ]
    all_query_runtime = sum(r.duration_milliseconds for r in all_query_runtimes) / len(
        all_query_runtimes
    )

    return objective_value, all_query_runtime


def get_fastest_slowest_device(input: TieringAlgorithmInput):
    devices = sort_devices(input.meta_segments, input.mappings, input.run_identifier)
    fastest_device = devices.iloc[[0]]
    slowest_device = devices.iloc[[len(devices) - 1]]
    logger.debug(f"Fastest device: {fastest_device}, slowest device: {slowest_device}")
    return fastest_device, slowest_device


def create_device_budget_input(original_input: TieringAlgorithmInput):
    input = copy(original_input)
    input.mappings = deepcopy(input.mappings)
    input.objective_mode = OBJECTIVE_MODE_DEVICE_BUDGET
    input.run_identifier = (
        f"{input.run_identifier}_runtime_objective_budget_interpolation"
    )
    return input


def measure_min_max_objective_latency(input: TieringAlgorithmInput):
    global ALL_QUERY_RUNTIME_MAX
    global ALL_QUERY_RUNTIME_MIN
    global ALL_QUERY_OBJECTIVE_MIN
    global ALL_QUERY_OBJECTIVE_MAX

    original_input = input
    input = create_device_budget_input(original_input)

    fastest_device, slowest_device = get_fastest_slowest_device(input)

    (
        objective_value_min,
        all_query_runtime_milliseconds_min,
    ) = get_objective_and_runtime_for_device_assignment(
        original_input,
        input,
        fastest_device,
        input.args.tiering_lp_runtime_budget_all_query_ms_min,
        ALL_QUERY_RUNTIME_MIN,
        ALL_QUERY_OBJECTIVE_MIN,
        (ALL_QUERY_RUNTIME_MIN is not None and ALL_QUERY_OBJECTIVE_MIN is not None),
    )
    ALL_QUERY_RUNTIME_MIN = all_query_runtime_milliseconds_min
    ALL_QUERY_OBJECTIVE_MIN = objective_value_min

    (
        objective_value_max,
        all_query_runtime_milliseconds_max,
    ) = get_objective_and_runtime_for_device_assignment(
        original_input,
        input,
        slowest_device,
        input.args.tiering_lp_runtime_budget_all_query_ms_max,
        ALL_QUERY_RUNTIME_MAX,
        ALL_QUERY_OBJECTIVE_MAX,
        (ALL_QUERY_RUNTIME_MAX is not None and ALL_QUERY_OBJECTIVE_MAX is not None),
    )
    ALL_QUERY_RUNTIME_MAX = all_query_runtime_milliseconds_max
    ALL_QUERY_OBJECTIVE_MAX = objective_value_max

    logger.debug(
        f"Objective value min: {objective_value_min}, max: {objective_value_max}"
    )
    logger.debug(
        f"All query runtime min: {all_query_runtime_milliseconds_min}, max: {all_query_runtime_milliseconds_max}"
    )

    with open(O2_CALIBRATION_FILE, "a") as f:
        f.write(
            f"{input.benchmark_config.server_config.scale_factor},{objective_value_min},{objective_value_max},{all_query_runtime_milliseconds_min},{all_query_runtime_milliseconds_max}"
        )

    return (
        objective_value_min,
        objective_value_max,
        all_query_runtime_milliseconds_min,
        all_query_runtime_milliseconds_max,
    )


def get_measured_min_max_objective_latency(input: TieringAlgorithmInput):
    assert input.args.tiering_lp_min_max_objective_calibration is not None
    df = pd.read_csv(input.args.tiering_lp_min_max_objective_calibration)
    m = df.iloc[0]
    if m["sf"] != input.spec.scale_factor:
        logger.warn(
            f"Scale factor {input.spec.scale_factor} but calibration uses {m['sf']}"
        )
    return (
        m["objective_value_min"],
        m["objective_value_max"],
        m["all_query_runtime_milliseconds_min"],
        m["all_query_runtime_milliseconds_max"],
    )


def interpolate_runtime_objective_budget(input: TieringAlgorithmInput):
    global ALL_QUERY_RUNTIME_MAX
    global ALL_QUERY_RUNTIME_MIN
    global ALL_QUERY_OBJECTIVE_MIN
    global ALL_QUERY_OBJECTIVE_MAX

    if not input.objective_mode == OBJECTIVE_MODE_RUNTIME_BUDGET:
        return 1.0

    assert input.benchmark_config is not None, "Benchmark config must be set"

    if input.args.calibration_min_max_objective_latency:
        measure_min_max_objective_latency(input)
        sys.exit(0)

    else:
        (
            objective_value_min,
            objective_value_max,
            all_query_runtime_milliseconds_min,
            all_query_runtime_milliseconds_max,
        ) = get_measured_min_max_objective_latency(input)

    logger.debug(f"Runtime percentage: {input.runtime_budget_percentage}")

    runtime_value_ms = lerp(
        all_query_runtime_milliseconds_min,
        all_query_runtime_milliseconds_max,
        input.runtime_budget_percentage,
    )

    if input.spec is not None:
        input.spec.runtime_budget_seconds = runtime_value_ms / 1000.0

    objective_value = lerp(
        objective_value_min,
        objective_value_max,
        input.runtime_budget_percentage
        ** input.args.tiering_lp_runtime_budget_objective_runtime_exponent,
    )
    logger.debug(f"Objective value: {objective_value}, runtime ms: {runtime_value_ms}")

    return objective_value


def get_tiering_config(
    model,
    table_ids_to_names: Dict[int, str],
    device_ids_to_names: Dict[int, str],
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    tiering_config = []

    set_segment_index(segments_df)

    for item in model.X:
        if round(model.X[item].value) == 1.0:
            try:
                # logger.info(f"Selected segment: {item}")
                (table_id, column_id, chunk_id, device_id) = item
                segment = segments_df.loc[table_id, column_id, chunk_id]
                tiering_config.append(
                    SegmentTieringAssignment(
                        table_id,
                        column_id,
                        chunk_id,
                        device_id,
                        table_ids_to_names[table_id],
                        device_ids_to_names[device_id],
                        model.SMC[table_id, column_id, chunk_id],
                        sum(int(segment[p]) for p in SEGMENT_ACCESS_PATTERNS),
                    )
                )
            except KeyError:
                logger.warning(f"KeyError: {item}")
                # column LP chooses non-existing segments
                continue
    df = pd.DataFrame(tiering_config)
    return df


def print_discrete_sizes(model):
    logger.debug("Discrete sizes:")
    for item in model.Y:
        if round(model.Y[item].value) == 1.0 and round(model.Z[item].value) == 1.0:
            (device, interval) = item
            logger.debug(f"{device} {interval} = 1.0")


def print_overview(result, model, device_ids_to_names: Dict[int, str]):

    solver_logs = result.json_repn()["Solver"][0]
    condition = solver_logs["Termination condition"]
    logger.info(
        f"Result: {condition}, finished in: {float(solver_logs['Wall time']):.4f} s"
    )
    logger.info(f"{solver_logs}")

    total_gb = 0
    for i in range(len(model.D)):
        if hasattr(model, "MemoryBudgetConstraint"):
            budget_gb = bytes_to_gb(1024**2 * model.DB[i].value)
            used_gb = bytes_to_gb(model.MemoryBudgetConstraint[i].body())
            total_gb += used_gb
            used_percentage = (
                ((used_gb / budget_gb)) if budget_gb > 0 else 1.0
            ) * 100.0
            logger.info(
                f"Device {i} budget: {(budget_gb):.3f}GB, used: {(used_gb):.3f}GB ({used_percentage:.2f}%) ({device_ids_to_names[i]})"
            )
        # todo other constraints
    logger.info(f"Aggregated size of all segments: {total_gb:.3f}GB")

    if condition != "optimal":
        logger.warning("Please check solution. Found solution not optimal.")


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
    mappings = input.mappings
    logger.debug(
        f"Creating LP model for {num_tables} tables, {num_columns} columns, {num_chunks} chunks."
    )

    model = pyo.ConcreteModel()

    # set of tables
    model.T = pyo.Set(initialize=range(0, num_tables))

    # set of columns
    model.M = pyo.Set(initialize=range(0, num_columns))

    # set of chunks
    model.N = pyo.Set(initialize=range(0, num_chunks))

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
    model.X = pyo.Var(model.T, model.M, model.N, model.D, within=pyo.Binary)

    # todo move all of this up
    for d in model.D:
        segment_sizes[f"cost_device_{d}"] /= 10**6

    SEGMENT_ID_VARS = ["table_id", "column_id", "chunk_id"]
    segments_df = segment_sizes.reset_index(drop=True)[
        SEGMENT_ID_VARS + ["size_in_bytes"] + [f"cost_device_{d}" for d in model.D]
    ]
    cost_cols = [c for c in segments_df if c.startswith("cost_device_")]
    segments_cost_df = segments_df.melt(
        id_vars=SEGMENT_ID_VARS,
        value_vars=cost_cols,
        value_name="cost",
    )
    segments_cost_df["device_id"] = (
        segments_cost_df["variable"].str.replace("cost_device_", "").astype(int)
    )
    cost_dict = segments_cost_df.set_index(SEGMENT_ID_VARS + ["device_id"])[
        "cost"
    ].to_dict()

    # logger.debug(f"Cost dict: {cost_dict}")

    model.C = pyo.Param(
        model.T,
        model.M,
        model.N,
        model.D,
        within=pyo.NonNegativeReals,
        initialize=cost_dict,
        mutable=False,
        default=0.0,
    )

    memory_dict = (
        segments_df[SEGMENT_ID_VARS + ["size_in_bytes"]]
        .set_index(SEGMENT_ID_VARS)["size_in_bytes"]
        .to_dict()
    )
    model.SMC = pyo.Param(
        model.T,
        model.M,
        model.N,
        within=pyo.NonNegativeIntegers,
        initialize=memory_dict,
        default=0,
        mutable=False,
    )

    def runtime(model):
        return sum(
            model.X[table_id, column_id, chunk_id, device_id]
            * model.C[table_id, column_id, chunk_id, device_id]
            for table_id in model.T
            for column_id in model.M
            for chunk_id in model.N
            for device_id in model.D
        )

    def device_memory_usage_byte(model, d):
        return sum(
            (
                model.X[table_id, column_id, chunk_id, d]
                * model.SMC[table_id, column_id, chunk_id]
            )
            for table_id in model.T
            for column_id in model.M
            for chunk_id in model.N
        )

    def dollar_cents_cost(model):
        costs = 0
        for d in model.D:
            continuous_size_bytes = device_memory_usage_byte(model, d)
            costs += model.DD[d] * continuous_size_bytes

        return costs / 10**9

    def memory_budget_rule(model, d):  # only per storage device d
        return (device_memory_usage_byte(model, d) / (1024**2)) <= model.DB[d]

    def dollar_budget_rule(model):
        return dollar_cents_cost(model) <= input.dollar_budget_cents

    def runtime_objective_budget_rule(model):
        return runtime(model) <= runtime_objective_budget * 1.001

    if input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET:
        model.Obj = pyo.Objective(rule=runtime)
        model.MemoryBudgetConstraint = pyo.Constraint(model.D, rule=memory_budget_rule)
    elif input.objective_mode == OBJECTIVE_MODE_DOLLAR_BUDGET:
        model.Obj = pyo.Objective(rule=runtime)
        if input.args.tiering_lp_use_discrete_device_capacities:
            add_discrete_dollar_cents_costs(model, input, segments_df)
        else:
            model.DollarBudgetConstraint = pyo.Constraint(rule=dollar_budget_rule)
    elif input.objective_mode == OBJECTIVE_MODE_RUNTIME_BUDGET:
        model.Obj = pyo.Objective(rule=dollar_cents_cost)
        # i = create_device_budget_input(input)
        # fastest_device, slowest_device = get_fastest_slowest_device(i)
        # logger.debug(f"Fastest device: {fastest_device}")
        # logger.debug(f"Slowest device: {slowest_device}")
        # _, obj = get_objective(input, i, fastest_device)
        # logger.debug(f"Objective fastest: {obj}")
        # _, obj = get_objective(input, i, slowest_device)
        # logger.debug(f"Objective slowest: {obj}")

        model.RuntimeObjectiveBudgetConstraint = pyo.Constraint(
            rule=runtime_objective_budget_rule
        )
    else:
        raise Exception(f"Unknown objective mode: {input.objective_mode}")

    if add_segment_selected_constraint:
        # existing segments must be selected, non-existing segments must not be selected
        def segment_selected_rule(model, t, m, n):
            return sum(model.X[t, m, n, device_id] for device_id in model.D) == int(
                (t, m, n) in segment_sizes.index
            )

        model.SegmentSelectedConstraint = pyo.Constraint(
            model.T, model.M, model.N, rule=segment_selected_rule
        )

    return model


def get_lp_model_params(input: TieringAlgorithmInput):
    lp_segments = input.meta_segments.copy()
    set_segment_index(lp_segments)
    num_tables = len(lp_segments["table_id"].unique())
    num_columns = lp_segments["column_id"].max() + 1
    num_chunks = lp_segments["chunk_id"].max() + 1
    device_budgets_in_MB = deepcopy(input.mappings.device_ids_to_storage_budgets_bytes)
    for k, v in device_budgets_in_MB.items():
        device_budgets_in_MB[k] = v / (1024**2)

    return (
        lp_segments,
        num_tables,
        num_columns,
        num_chunks,
        device_budgets_in_MB,
    )


def write_segment_cost_file_header(run_identifier: str):
    with open(TIERING_CONFIG_DIR / f"{run_identifier}_lp_segment_costs.csv", "w") as f:
        f.write("table_id,column_id,chunk_id,device_id,cost\n")


class DetermineTieringLP(TieringAlgorithm):
    def __init__(self, input: TieringAlgorithmInput):
        super().__init__(input)

    def set_up_solver(self):
        input = self.input

        write_segment_cost_file_header(input.run_identifier)

        # reduce numeric range (should be < 10^9 for gurobi, see https://www.gurobi.com/documentation/9.5/refman/does_my_model_have_numeric.html)

        (
            lp_segments,
            num_tables,
            num_columns,
            num_chunks,
            device_budgets_in_MB,
        ) = get_lp_model_params(input)

        runtime_objective_budget = interpolate_runtime_objective_budget(input)

        self.model = create_lp_model(
            num_tables,
            num_columns,
            num_chunks,
            device_budgets_in_MB,
            lp_segments,
            input,
            runtime_objective_budget,
        )

    @timed
    def _solve(self):
        solver = SolverFactory(
            "gurobi",
        )
        solver.options["threads"] = 8
        solver.options["timeLimit"] = 500  # seconds
        solver.options["MIPGap"] = 0.01  # percent
        logger.info("Solving LP model...")
        self.result = solver.solve(self.model, tee=True)
        if os.path.exists("./recording000.grbr"):
            shutil.move(
                "./recording000.grbr",
                TIERING_CONFIG_DIR / f"{self.input.run_identifier}_recording.grbr",
            )

    def solve(self):
        res = self._solve()
        self.runtime_ms = res.runtime_milliseconds

    def get_solver_result(self) -> TieringAlgorithmResult:
        mappings = self.input.mappings

        try:
            print_overview(self.result, self.model, mappings.device_ids_to_names)
            if self.input.args.tiering_lp_use_discrete_device_capacities:
                print_discrete_sizes(self.model)
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Failed to print overview: {e}")

        tiering_config = get_tiering_config(
            self.model,
            mappings.table_ids_to_names,
            mappings.device_ids_to_names,
            self.input.meta_segments,
        )
        return (
            tiering_config,
            mappings.device_ids_to_names,
            mappings.table_ids_to_names,
            self.model.Obj.expr(),
            True,
            self.runtime_ms,
        )
