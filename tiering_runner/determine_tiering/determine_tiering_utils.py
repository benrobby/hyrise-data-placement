import logging
from typing import Dict, List

import pandas as pd
from tiering_runner.determine_tiering.determine_tiering_greedy_utils import \
    sort_devices
from tiering_runner.helpers.globals import (OBJECTIVE_MODE_DEVICE_BUDGET,
                                            OBJECTIVE_MODE_DOLLAR_BUDGET,
                                            OBJECTIVE_MODE_RUNTIME_BUDGET)
from tiering_runner.helpers.timing import TimedReturnT, timed
from tiering_runner.helpers.types import (DeviceCalibrationResults,
                                          QueryResultT, TieringAlgorithmInput,
                                          TieringAlgorithmMappings,
                                          TieringAlgorithmResult,
                                          TieringDevice)

logger = logging.getLogger("determine_tiering")


def clamp(min_value, max_value, x):
    return min(max(x, min_value), max_value)


def lerp(a, b, p):
    return a + (b - a) * p


def get_mappings(
    meta_segments: QueryResultT,
    devices: List[TieringDevice],
    device_ids_and_datatypes_to_calibration_results: DeviceCalibrationResults,
):
    device_names_to_ids = {d.device_name: d.id for d in devices}
    device_ids_to_names = {i: d for d, i in device_names_to_ids.items()}
    device_ids_to_storage_budgets_bytes = {d.id: d.capacity_bytes() for d in devices}
    device_ids_to_discrete_capacity_option_bytes = {
        d.id: d.discrete_capacity_option_bytes for d in devices
    }
    device_ids_to_cents_per_GB = {d.id: d.dollar_cents_per_GB for d in devices}
    table_names_to_ids = {
        t: i for i, t in enumerate(meta_segments["table_name"].unique())
    }
    table_ids_to_names = {i: t for t, i in table_names_to_ids.items()}
    device_calibration_ids_to_ids = { d.get_calibration_id(): d.id for d in devices}

    return TieringAlgorithmMappings(
        device_names_to_ids,
        device_ids_to_names,
        device_ids_to_storage_budgets_bytes,
        {
            k: v
            for k, v in device_ids_and_datatypes_to_calibration_results.items()
            if int(k.split("_")[0]) in device_calibration_ids_to_ids.keys()
        },
        device_ids_to_cents_per_GB,
        device_ids_to_discrete_capacity_option_bytes,
        table_names_to_ids,
        table_ids_to_names,
        device_calibration_ids_to_ids,
    )


def set_segment_index(df: pd.DataFrame):
    df.set_index(
        pd.MultiIndex.from_frame(
            df[["table_id", "column_id", "chunk_id"]],
            names=["table_id", "column_id", "chunk_id"],
        ),
        inplace=True,
    )


def bytes_to_gb(b):
    return b / (1024**3)


def print_config(
    tiering_config: pd.DataFrame,
    device_ids_to_names: Dict[int, str],
    table_ids_to_names: Dict[int, str],
):
    set_segment_index(tiering_config)
    tiering_config.sort_index(inplace=True)

    logger.info(f"Resulting tiering config:\n{tiering_config}")
    logger.info(f"Device ids are: {device_ids_to_names}")

    for table_id, table_name in table_ids_to_names.items():

        log = ""
        num_chunks = tiering_config.loc[table_id]["chunk_id"].max() + 1

        for chunk_id in range(num_chunks):
            chunk_segment_devices = tiering_config.loc[table_id, :, chunk_id][
                "device_id"
            ].astype(str)
            log += " ".join(chunk_segment_devices) + "\n"

        log = log.strip("\n")
        if log != "":
            logger.info(f"Table {table_id} ({table_name}) tiering assignment:\n{log}")


def get_used_bytes_per_device(tiering_config: pd.DataFrame):
    used_bytes_per_device = tiering_config.groupby("device_id").sum()[
        "segment_size_bytes"
    ]
    logger.debug(used_bytes_per_device)
    return used_bytes_per_device


def print_solution_overview(
    optimization_method: str,
    res: TimedReturnT[TieringAlgorithmResult],
    mappings: TieringAlgorithmMappings,
):
    (
        tiering_config,
        device_ids_to_names,
        _,
        objective_value_for_entire_config,
        _,
        _,
    ) = res.result

    used_bytes_per_device = get_used_bytes_per_device(tiering_config)

    logger.info(
        f"Determine Tiering ({optimization_method}) finished in: {(res.runtime_milliseconds / 1000):.4f} s"
    )

    total_gb = 0
    for device_id, device_name in device_ids_to_names.items():
        budget_gb = bytes_to_gb(mappings.device_ids_to_storage_budgets_bytes[device_id])
        used_gb = bytes_to_gb(
            used_bytes_per_device.loc[device_id]
            if device_id in used_bytes_per_device
            else 0
        )
        total_gb += used_gb
        used_percentage = (((used_gb / budget_gb)) if budget_gb > 0 else 1.0) * 100.0
        logger.info(
            f"Device {device_id} {device_name} budget: {(budget_gb):.3f}GB, used: {(used_gb):.3f}GB ({used_percentage:.2f}%)"
        )

    logger.info(f"Aggregated size of all segments: {total_gb:.3f}GB")
    logger.info(f"Objective: {objective_value_for_entire_config:.3f}")


def assign_to_cheapest_device(
    input: TieringAlgorithmInput,
    tiering_config: pd.DataFrame,
    segments_without_accesses_indices: pd.DataFrame,
    segments_without_accesses_size: float,
):
    mappings = input.mappings
    devices_sorted = sort_devices(input.meta_segments, mappings, input.run_identifier)
    devices_sorted = devices_sorted.sort_values(
        by="dollar_cents_per_GB", ascending=True
    )
    cheapest_device = devices_sorted.iloc[[0]]
    logger.debug(f"cheapest device: {cheapest_device}")
    device_id = int(cheapest_device["device_id"])
    device_name = mappings.device_ids_to_names[device_id]

    logger.info(
        f"Assigning all segments without access ({len(segments_without_accesses_indices)} of {len(tiering_config)}, {segments_without_accesses_size}) to device {device_id} {device_name}"
    )
    tiering_config.loc[segments_without_accesses_indices, "device_id"] = device_id
    tiering_config.loc[segments_without_accesses_indices, "device_name"] = device_name


def get_device_info(device, mappings):
    device_id = int(device["device_id"])
    budget_bytes = mappings.device_ids_to_storage_budgets_bytes[device_id]
    used_bytes = int(device["used_bytes"])
    return device_id, budget_bytes, used_bytes


def postprocess_tiering_conf_device_budget(
    tiering_config: pd.DataFrame,
    input: TieringAlgorithmInput,
    segments_without_accesses_indices: pd.DataFrame,
):
    mappings = input.mappings

    # tiering_config.to_csv(
    #     TIERING_CONFIG_DIR / f"{input.run_identifier}_tiering_config_with_all.csv"
    # )

    # logger.debug(f"tiering config: {tiering_config}\n{tiering_config.columns}")
    tiering_config.loc[segments_without_accesses_indices, "device_id"] = -1
    tiering_config.loc[segments_without_accesses_indices, "device_name"] = "unassigned"

    # tiering_config.to_csv(
    #     TIERING_CONFIG_DIR
    #     / f"{input.run_identifier}_tiering_config_with_unassigned.csv"
    # )

    devices_sorted = sort_devices(input.meta_segments, mappings, input.run_identifier)
    used_bytes_per_device = tiering_config.groupby("device_id").sum()
    devices_sorted.set_index("device_id", inplace=True)

    logger.debug(f"devices sorted: \n{devices_sorted}\n{devices_sorted.columns}")
    logger.debug(
        f"used_bytes_per_device: \n{used_bytes_per_device}\n{used_bytes_per_device.columns}"
    )

    for device_id in input.mappings.device_ids_to_names.keys():
        used_bytes = 0
        try:
            used_bytes = used_bytes_per_device.loc[device_id]["segment_size_bytes"]
        except KeyError:
            pass
        devices_sorted.loc[device_id, "used_bytes"] = used_bytes

    devices_sorted.reset_index(inplace=True, drop=False)
    logger.debug(f"devices sorted: \n{devices_sorted}\n{devices_sorted.columns}")

    current_device_index = 0

    set_segment_index(tiering_config)

    remaining_segments = tiering_config[tiering_config["device_id"] == -1]
    remaining_segments.sort_values(
        by="segment_size_bytes", inplace=True, ascending=True
    )

    logger.debug(
        f'Postprocessing segments without accesses ({len(remaining_segments)}, {remaining_segments["segment_size_bytes"].sum()})'
    )

    while len(remaining_segments) > 0:
        logger.debug(f"Filling device {current_device_index}")
        assert current_device_index < len(
            devices_sorted
        ), "no device has enough free space to fit all segments without accesses"

        device = devices_sorted.iloc[[current_device_index]]
        device_id, budget_bytes, used_bytes = get_device_info(device, mappings)
        device_name = mappings.device_ids_to_names[device_id]
        logger.debug(
            f"Device {current_device_index} {device_name} has budget {budget_bytes} and usage {used_bytes}"
        )

        for _, segment in remaining_segments.iterrows():

            device_used_bytes = int(device["used_bytes"])
            segment_size_bytes = int(segment["segment_size_bytes"])
            table_id = int(segment["table_id"])
            column_id = int(segment["column_id"])
            chunk_id = int(segment["chunk_id"])

            if device_used_bytes + segment_size_bytes > budget_bytes:
                logger.debug(
                    f"Device {current_device_index} with id {device_id} is full: {device_used_bytes}, {budget_bytes}, {segment_size_bytes}, going to next one"
                )
                break

            # logger.debug(
            #     f"Assigning segment {table_id}:{column_id}:{chunk_id} to device {device_id}"
            # )

            tiering_config.loc[(table_id, column_id, chunk_id), "device_id"] = device_id
            tiering_config.loc[
                (table_id, column_id, chunk_id), "device_name"
            ] = device_name

            devices_sorted.iloc[
                current_device_index,
                devices_sorted.columns.get_loc("used_bytes"),
            ] = (
                device_used_bytes + segment_size_bytes
            )
            device = devices_sorted.iloc[[current_device_index]]

        logger.debug(
            f'Finished filling device {current_device_index} {device_name} with budget {budget_bytes} and usage {int(device["used_bytes"])}'
        )

        remaining_segments = tiering_config[tiering_config["device_id"] == -1]
        remaining_segments.sort_values(
            by="segment_size_bytes", inplace=True, ascending=True
        )
        current_device_index += 1

        logger.debug(
            f"Finished filling current device, switching to the next. Next device: {current_device_index}, remaining segments: {len(remaining_segments)}"
        )

    tiering_config.reset_index(inplace=True, drop=True)
    return tiering_config


def postprocess_tiering_conf(
    tiering_config: pd.DataFrame,
    input: TieringAlgorithmInput,
) -> pd.DataFrame:

    if not input.args.tiering_postprocess:
        logger.warn("Not postprocessing tiering config")
        return tiering_config

    segments_without_accesses_indices = tiering_config["access_count_sum"] == 0
    segments_without_accesses = tiering_config[segments_without_accesses_indices]
    segments_without_accesses_size = segments_without_accesses[
        "segment_size_bytes"
    ].sum()
    logger.debug(f"segments_without_accesses_size: {segments_without_accesses_size}")
    logger.debug(
        f"segments without accesses: ({len(segments_without_accesses)} of {len(tiering_config)})"
    )

    if input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET:
        return postprocess_tiering_conf_device_budget(
            tiering_config, input, segments_without_accesses_indices
        )
    elif input.objective_mode == OBJECTIVE_MODE_DOLLAR_BUDGET:
        # find segments without accesses
        # find cheapest device
        # assign all segments without accesses to this device. We can not exceed the budget with that
        # because they segments were already assigned to a device that was at least as expensive as the cheapest one.

        assign_to_cheapest_device(
            input,
            tiering_config,
            segments_without_accesses_indices,
            segments_without_accesses_size,
        )
        return tiering_config
    elif input.objective_mode == OBJECTIVE_MODE_RUNTIME_BUDGET:
        # assign to cheapest device, this still minimizes the $ cost (our objective)
        # the runtime objective budget can not be violated since the costs will be zero anyways.
        assign_to_cheapest_device(
            input,
            tiering_config,
            segments_without_accesses_indices,
            segments_without_accesses_size,
        )
        return tiering_config
    else:
        assert False, f"objective mode {input.objective_mode} not implemented"


def calculate_cost_predictions(
    tiering_config: pd.DataFrame, input: TieringAlgorithmInput
):
    tiering_config["segment_device_cost"] = 0
    mappings = input.mappings
    meta_segments = input.meta_segments.copy()
    meta_segments.set_index(["table_id", "column_id", "chunk_id"], inplace=True)

    assert set(tiering_config["device_id"].unique()).issubset(
        set(mappings.device_ids_to_names.keys())
    )

    # tiering_config.to_csv("tiering_config.csv")
    # meta_segments.to_csv("meta_segments.csv")

    def f(x):
        # print(x)
        return meta_segments.loc[
            (x["table_id"], x["column_id"], x["chunk_id"]),
            f'cost_device_{x["device_id"]}',
        ]

    tiering_config["assigned_segment_cost"] = tiering_config.apply(
        f,
        axis=1,
    )

    return tiering_config


@timed
def execute_tiering_algorithm(
    algorithm_cls, input: TieringAlgorithmInput
) -> TieringAlgorithmResult:
    algorithm_obj = algorithm_cls(input)
    algorithm_obj.set_up_solver()
    algorithm_obj.solve()
    return algorithm_obj.get_solver_result()
