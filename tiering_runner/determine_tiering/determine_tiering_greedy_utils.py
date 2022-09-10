import logging

import pandas as pd
from tiering_runner.helpers.globals import (SEGMENT_ACCESS_PATTERNS,
                                            TIERING_CONFIG_DIR)
from tiering_runner.helpers.types import TieringAlgorithmMappings

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


def sort_devices(
    segments_df: pd.DataFrame,
    mappings: TieringAlgorithmMappings,
    run_identifier: str,
):
    # compute weights for access patterns for workload
    device_df = pd.DataFrame(list(mappings.device_ids_to_calibration_results.values()))
    device_df["device_calibration_id"] = device_df["device_id"]
    device_df["device_id"] = device_df["device_calibration_id"].map(mappings.device_calibration_ids_to_ids)
    device_df["device_name"] = device_df["device_id"].map(mappings.device_ids_to_names)
    device_df["dollar_cents_per_GB"] = device_df["device_id"].map(
        mappings.device_ids_to_dollar_cents_per_GB
    )

    access_pattern_usage_counts = [
        segments_df[p].sum() for p in SEGMENT_ACCESS_PATTERNS
    ]
    # score devices based on sum_i (access_pattern_calibration_i * access_pattern_usage_i)
    device_df["runtime_performance"] = (
        device_df[APD[0]] * access_pattern_usage_counts[0]
        + device_df[APD[1]] * access_pattern_usage_counts[1]
        + device_df[APD[2]] * access_pattern_usage_counts[2]
        + device_df[APD[3]] * access_pattern_usage_counts[3]
    )

    logger.debug(f"device_df: {device_df.columns} {device_df}")

    device_df_base = device_df[device_df["datatype"] == "float"]
    device_df_string = device_df[device_df["datatype"] == "string"]
    device_df_m = pd.merge(
        device_df_base, device_df_string, on="device_id", suffixes=("_float", "_string")
    )

    device_df_m.to_csv(
        TIERING_CONFIG_DIR / f"{run_identifier}_merged_greedy_devices_sorted.csv"
    )

    for p in ALL_APD:
        device_df_m[p + "_mean"] = (
            device_df_m[p + "_float"] + device_df_m[p + "_string"]
        ) / 2.0
    device_df_m["runtime_performance_mean"] = (
        device_df_m["runtime_performance_string"]
        + device_df_m["runtime_performance_float"]
    ) / 2.0
    device_df_m["dollar_cents_per_GB"] = device_df_m["dollar_cents_per_GB_float"]

    device_df_m.to_csv(
        TIERING_CONFIG_DIR / f"{run_identifier}_merged_1_greedy_devices_sorted.csv"
    )

    # sort devices by score
    devices_sorted = device_df_m.sort_values(
        by="random_multiple_chunk_accesses_runtime_float", ascending=True
    )
    devices_sorted["used_bytes"] = 0
    devices_sorted.to_csv(
        TIERING_CONFIG_DIR / f"{run_identifier}_greedy_devices_sorted.csv"
    )
    return devices_sorted
