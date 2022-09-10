import logging
import time
import tracemalloc
from pathlib import Path
from typing import List

import pandas as pd
from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.cost_model.get_cost_model import get_cost_model
from tiering_runner.determine_tiering.determine_tiering_column_lp import (
    DetermineTieringMosaicLP,
)
from tiering_runner.determine_tiering.determine_tiering_dram import DetermineTieringDram
from tiering_runner.determine_tiering.determine_tiering_greedy import (
    DetermineTieringGreedy,
)
from tiering_runner.determine_tiering.determine_tiering_knapsack import (
    DetermineTieringKnapsack,
)
from tiering_runner.determine_tiering.determine_tiering_lp import DetermineTieringLP
from tiering_runner.determine_tiering.determine_tiering_utils import (
    calculate_cost_predictions,
    execute_tiering_algorithm,
    get_mappings,
    postprocess_tiering_conf,
    print_solution_overview,
)
from tiering_runner.helpers.globals import (
    OPTIMIZATION_METHOD_COLUMN_LP,
    OPTIMIZATION_METHOD_DRAM,
    OPTIMIZATION_METHOD_GREEDY,
    OPTIMIZATION_METHOD_KNAPSACK,
    OPTIMIZATION_METHOD_LP,
    TIERING_CONFIG_DIR,
    TIERING_DEVICES_META_CSV_FILE,
    TIERING_RUNS_META_CSV_FILE,
)
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import (
    BenchmarkConfig,
    DetermineTieringRunMetadata,
    DeviceCalibrationResults,
    TieringAlgorithmInput,
    TieringDevice,
)
from tiering_runner.hyrise_server.hyrise_server import HyriseServer
from tiering_runner.run_calibration.run_calibration import calibration_df_to_dict

logger = logging.getLogger("determine_tiering")


def append_tiering_meta_info(
    meta_input: DetermineTieringRunMetadata,
    optimization_method: str,
    cost_model_name: str,
    devices: List[TieringDevice],
    tiering_config: pd.DataFrame,
    objective_value_for_entire_config: float,
):
    logger.debug("appending tiering meta info")
    with open(TIERING_DEVICES_META_CSV_FILE, "a") as tiering_meta_csv:
        tiering_run_id = f"{meta_input.benchmark_name}_{meta_input.scale_factor}_{optimization_method}_{time.time()}"
        for device in devices:
            device_used_bytes = tiering_config[
                tiering_config["device_id"] == device.id
            ]["segment_size_bytes"].sum()
            tiering_meta_csv.write(
                f"{meta_input.run_id},{meta_input.run_sorted_id},{tiering_run_id},{meta_input.benchmark_name},{meta_input.scale_factor},{optimization_method},{cost_model_name},{device.device_name},{device.id},{device.capacity_bytes()},{device_used_bytes},{device.used_bytes_percentage(device_used_bytes)},{objective_value_for_entire_config}\n"
            )


def write_tiering_config(tiering_config, run_identifier):
    tiering_config.to_csv(
        TIERING_CONFIG_DIR / f"{run_identifier}_tiering_config.csv",
        sep=",",
    )


optimization_methods_to_algorithms = {
    OPTIMIZATION_METHOD_LP: DetermineTieringLP,
    OPTIMIZATION_METHOD_COLUMN_LP: DetermineTieringMosaicLP,
    OPTIMIZATION_METHOD_KNAPSACK: DetermineTieringKnapsack,
    OPTIMIZATION_METHOD_DRAM: DetermineTieringDram,
    OPTIMIZATION_METHOD_GREEDY: DetermineTieringGreedy,
}


@timed
def _determine_tiering_for_optimization_method(
    optimization_method: str,
    input: TieringAlgorithmInput,
):
    assert (
        optimization_method in optimization_methods_to_algorithms.keys()
    ), f"optimization method {optimization_method} not implemented"

    algorithm = optimization_methods_to_algorithms[optimization_method]

    tracemalloc.start()
    tracemalloc.reset_peak()
    old_memory_size, old_memory_peak = tracemalloc.get_traced_memory()

    res = execute_tiering_algorithm(algorithm, input)

    current_memory_size, current_memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert (
        current_memory_peak > old_memory_peak
    ), f"expecting memory usage to increase but was {current_memory_peak} {old_memory_peak}"

    logger.debug(
        f"Memory usage: {old_memory_size} {old_memory_peak} {current_memory_size} {current_memory_peak}"
    )

    spec = input.spec
    solver_runtime = res.result[5]
    e2e_runtime = res.runtime_milliseconds
    solver_peak_memory_bytes = 0
    e2e_peak_memory_bytes = current_memory_peak - old_memory_peak

    if spec is not None:
        with open(TIERING_RUNS_META_CSV_FILE, "a") as f:
            f.write(
                f"{spec.to_csv_row()},{solver_runtime},{e2e_runtime},{solver_peak_memory_bytes},{e2e_peak_memory_bytes}\n"
            )

    return res.result


@timed
def _read_tiering_file(tiering_config_file_path: str, run_identifier, mappings):
    assert Path(
        tiering_config_file_path
    ).exists(), f"{tiering_config_file_path} does not exist"
    tiering_config = pd.read_csv(tiering_config_file_path)
    write_tiering_config(tiering_config, run_identifier)
    return (
        tiering_config,
        mappings.device_ids_to_names,
        mappings.table_ids_to_names,
        0,
        True,
    )


def _determine_tiering_or_read_file(
    optimization_method: str,
    input: TieringAlgorithmInput,
    tiering_config_file_path: str = None,
):

    if tiering_config_file_path is not None and tiering_config_file_path != "":
        logging.info(
            f"Not running optimization algorithm, using tiering config from file {tiering_config_file_path}"
        )
        return _read_tiering_file(
            tiering_config_file_path, input.run_identifier, input.mappings
        )
    else:
        logging.info(f"determining tiering config with {optimization_method}")
        logger.debug(input.mappings)
        return _determine_tiering_for_optimization_method(optimization_method, input)


def determine_tiering(
    server: HyriseServer,
    benchmark_config: BenchmarkConfig,
    tiering_run_metadata: DetermineTieringRunMetadata,
    meta_segments: pd.DataFrame,
    calibration_results: DeviceCalibrationResults,
    optimization_method: str,
    devices: List[TieringDevice],
    args,
    cost_model_name: str,
    objective_mode: str,
    dollar_budget_cents: float,
    runtime_budget_percentage: float,
    tiering_config_file_path: str = None,
    spec: TieringBenchmarkSpec = None,
) -> pd.DataFrame:

    mappings = get_mappings(meta_segments, devices, calibration_results)
    cost_model = get_cost_model(args, cost_model_name)

    for device in devices:
        # logging.debug(f"device {device.id} {device.device_name} {device.get_calibration_id()}")
        # logging.debug("device_ids_to_calibration_results")
        # logging.debug(mappings.device_ids_to_calibration_results)
        calibration_for_device = {
            c.datatype: [
                c.sequential_accesses_runtime,
                c.random_accesses_runtime,
                c.monotonic_accesses_runtime,
                c.single_point_accesses_runtime,
            ]
            for c in mappings.device_ids_to_calibration_results.values()
            if c.device_id == device.get_calibration_id()
        }
        # logging.debug(f"calibration_for_device {calibration_for_device}")

        meta_segments[f"cost_device_{device.id}"] = meta_segments.apply(
            lambda x: cost_model.get_cost(x, calibration_for_device, device),
            axis=1,
        )
    meta_segments.to_csv(
        TIERING_CONFIG_DIR / f"{spec.id if spec is not None else 0}_meta_segments.csv",
        sep=",",
    )

    input = TieringAlgorithmInput(
        meta_segments,
        mappings,
        tiering_run_metadata.run_identifier,
        cost_model,
        objective_mode,
        dollar_budget_cents * (spec.scale_factor_multiplier if spec is not None else 1),
        runtime_budget_percentage,
        server,
        benchmark_config,
        args,
        spec,
    )

    res = _determine_tiering_or_read_file(
        optimization_method, input, tiering_config_file_path
    )

    (
        tiering_config,
        device_ids_to_names,
        table_ids_to_names,
        objective_value_for_entire_config,
        should_postprocess,
        solver_runtime_ms,
    ) = res.result

    if should_postprocess:
        tiering_config = postprocess_tiering_conf(tiering_config, input)

    tiering_config = calculate_cost_predictions(tiering_config, input)

    print_solution_overview(optimization_method, res, mappings)
    # print_config(tiering_config, device_ids_to_names, table_ids_to_names)
    append_tiering_meta_info(
        tiering_run_metadata,
        optimization_method,
        cost_model_name,
        devices,
        tiering_config,
        objective_value_for_entire_config,
    )

    write_tiering_config(tiering_config, input.run_identifier)

    assert len(tiering_config) == len(
        meta_segments
    ), f"{len(tiering_config)} != {len(meta_segments)}"

    return tiering_config


def determine_tiering_for_spec(
    server: HyriseServer,
    benchmark_config: BenchmarkConfig,
    spec: TieringBenchmarkSpec,
    meta_segments: pd.DataFrame,
    calibration_results: DeviceCalibrationResults,
    args,
) -> pd.DataFrame:
    run_metadata = DetermineTieringRunMetadata(
        spec.benchmark_name,
        spec.m_scale_factor(),
        str(spec.id),
        str(spec.sorted_id),
        f"{spec.id}_spec",
    )

    if spec.calibration_data_to_use is not None:
        logger.info(
            f"Using calibration data from {spec.calibration_data_to_use} for this spec"
        )
        df = pd.read_csv(spec.calibration_data_to_use)
        df.to_csv(
            TIERING_CONFIG_DIR / f"{run_metadata.run_identifier}_calibration.csv",
            sep=",",
        )
        calibration_results = calibration_df_to_dict(df)

    return determine_tiering(
        server,
        benchmark_config,
        run_metadata,
        meta_segments,
        calibration_results,
        spec.optimization_method,
        spec.m_devices(),
        args,
        spec.cost_model,
        spec.objective_mode,
        spec.dollar_budget_cents,
        spec.runtime_percentage,
        tiering_config_file_path=spec.tiering_config_to_use,
        spec=spec,
    )
