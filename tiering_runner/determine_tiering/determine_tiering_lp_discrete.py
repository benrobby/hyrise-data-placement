import logging

import numpy as np
import pyomo.environ as pyo
from tiering_runner.helpers.types import TieringAlgorithmInput

logger = logging.getLogger(__name__)


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


def add_discrete_dollar_cents_costs(model, input: TieringAlgorithmInput, segments_df):
    discrete_device_sizes_bytes = (
        input.mappings.device_ids_to_discrete_capacity_option_bytes
    )

    for k, v in discrete_device_sizes_bytes.items():
        if not (0 in v or 0.0 in v):
            v.append(0)
        if not (1 in v or 1.0 in v):
            v.append(1)

        discrete_device_sizes_bytes[k] = sorted(v)

    logger.debug(f"Discrete device sizes: {discrete_device_sizes_bytes}")

    num_discrete_capacities = len(discrete_device_sizes_bytes[0])
    for v in discrete_device_sizes_bytes.values():
        assert (
            len(v) == num_discrete_capacities
        ), f"Discrete device sizes must be the same length {len(v)} {num_discrete_capacities}"
    NUM_INTERVALS = num_discrete_capacities - 1
    logger.debug(f"NUM_INTERVALS: {NUM_INTERVALS}")

    model.CAP_INTERVALS = pyo.Set(initialize=range(0, NUM_INTERVALS))
    model.Z = pyo.Var(model.D, model.CAP_INTERVALS, within=pyo.Binary)
    model.Y = pyo.Var(model.D, model.CAP_INTERVALS, within=pyo.Binary)

    logger.debug(f"input.dollar_budget_cents: {input.dollar_budget_cents}")

    M = segments_df["size_in_bytes"].sum()
    logger.debug(f"Total memory usage: {M}")
    largest_device_size_bytes = max(
        cap
        for caps_per_device in discrete_device_sizes_bytes.values()
        for cap in caps_per_device
    )
    logger.debug(f"Largest device size: {largest_device_size_bytes}")
    M = 100 * max(
        M, largest_device_size_bytes
    )  # must be big enough, otherwise solver fails with infeasible

    NUM_DEVICES = len(input.mappings.device_ids_to_storage_budgets_bytes)
    logger.debug(f"NUM_DEVICES: {NUM_DEVICES}")

    model.ZYConstraintExactlyOneIntervalPerDevice = pyo.Constraint(
        model.D,
        rule=lambda model, d: sum(
            model.Z[d, i] * model.Y[d, i] for i in model.CAP_INTERVALS
        )
        == 1,
    )

    model.ZYConstraintEachDeviceHasInterval = pyo.Constraint(
        rule=lambda model: sum(
            model.Z[d, i] * model.Y[d, i] for d in model.D for i in model.CAP_INTERVALS
        )
        == NUM_DEVICES,
    )

    # device memory usage <= upper bound
    # Z

    model.ZConstraint1 = pyo.Constraint(
        model.D,
        model.CAP_INTERVALS,
        rule=lambda model, d, i: (
            discrete_device_sizes_bytes[d][i + 1] - device_memory_usage_byte(model, d)
            <= M * model.Z[d, i]
        ),
    )

    model.ZConstraint2 = pyo.Constraint(
        model.D,
        model.CAP_INTERVALS,
        rule=lambda model, d, i: (
            device_memory_usage_byte(model, d) - discrete_device_sizes_bytes[d][i + 1]
            <= M * (1 - model.Z[d, i])
        ),
    )

    # lower bound <= device memory usage
    # Y
    model.YConstraint1 = pyo.Constraint(
        model.D,
        model.CAP_INTERVALS,
        rule=lambda model, d, i: (
            device_memory_usage_byte(model, d) - discrete_device_sizes_bytes[d][i]
            <= M * model.Y[d, i]
        ),
    )
    model.YConstraint2 = pyo.Constraint(
        model.D,
        model.CAP_INTERVALS,
        rule=lambda model, d, i: (
            discrete_device_sizes_bytes[d][i] - device_memory_usage_byte(model, d)
            <= M * (1 - model.Y[d, i])
        ),
    )

    def dollar_budget_rule(model):
        return input.dollar_budget_cents >= sum(
            model.DD[d]
            * sum(
                model.Y[d, i]
                * model.Z[d, i]
                * discrete_device_sizes_bytes[d][i + 1]
                / 10**9  # upper bound
                for i in model.CAP_INTERVALS
            )
            for d in model.D
        )

    model.DollarBudgetConstraint = pyo.Constraint(rule=dollar_budget_rule)
