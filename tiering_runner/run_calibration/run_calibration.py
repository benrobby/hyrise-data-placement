import logging
from typing import Dict, Set, Tuple

import pandas as pd
from dacite import from_dict
from tiering_runner.helpers.globals import (CALIBRATION_FILE,
                                            DRAM_DEVICE,
                                            hyrise_device_name_to_device_name)
from tiering_runner.helpers.types import (DeviceCalibrationResult,
                                          DeviceCalibrationResults,
                                          HyriseServerConfig)
from tiering_runner.hyrise_server.hyrise_plugin_interface import \
    HyrisePluginInterface
from tiering_runner.hyrise_server.hyrise_server import HyriseServer

CALIBRATION_DF_RUNTIME_COLS = [
    "sequential_accesses_runtime",
    "random_single_chunk_accesses_runtime",
    "random_multiple_chunk_accesses_runtime",
    "random_accesses_runtime",
    "monotonic_accesses_runtime",
    "single_point_accesses_runtime",
]


def calibration_data_to_df(calibration_data: Dict[int, DeviceCalibrationResult]):
    df = pd.DataFrame.from_dict(
        {
            k: [
                v.sequential_accesses_runtime,
                v.random_single_chunk_accesses_runtime,
                v.random_multiple_chunk_accesses_runtime,
                v.random_accesses_runtime,
                v.monotonic_accesses_runtime,
                v.single_point_accesses_runtime,
                v.device_id,
                v.device_name,
                v.datatype,
            ]
            for k, v in calibration_data.items()
        },
        columns=(
            CALIBRATION_DF_RUNTIME_COLS + ["device_id", "device_name", "datatype"]
        ),
        orient="index",
    )
    return df


def calibration_df_to_dict(df: pd.DataFrame):
    return {
        f"{d['device_id']}_{d['datatype']}": from_dict(
            data_class=DeviceCalibrationResult, data=d
        )
        for d in df.to_dict("records")
    }


def get_static_calibration_data(all_devices, args):

    dram_speedup_factor = args.calibration_static_dram_speedup_factor
    logging.info(f"Using static calibration data with factor {dram_speedup_factor}")
    calibration_data = {}

    for device in all_devices:
        if device[0] == DRAM_DEVICE:
            runtime = 1.0
        else:
            runtime = dram_speedup_factor
        calibration_data[device[1] + "_float"] = DeviceCalibrationResult(
            runtime,
            runtime / 2.0,
            runtime / 2.0,
            runtime,
            runtime,
            runtime,
            device[1],
            device[0],
            "float",
        )
        calibration_data[device[1] + "_string"] = DeviceCalibrationResult(
            runtime,
            runtime / 2.0,
            runtime / 2.0,
            runtime,
            runtime,
            runtime,
            device[1],
            device[0],
            "string",
        )
    return calibration_data_to_df(calibration_data)


def get_measured_calibration_data(
    all_devices: Set[Tuple[str, int]], server_config: HyriseServerConfig, args
) -> DeviceCalibrationResults:
    logging.info("Running calibration queries in Hyrise")
    device_names_to_ids = {
        device_name: device_id for device_name, device_id in all_devices
    }

    with HyriseServer(server_config) as server:
        client = HyrisePluginInterface(server)
        client.set_devices(args.umap_buf_size_bytes, args.all_device_names)

        res = client.run_calibration(
            args.calibration_scale_factor,
            args.calibration_benchmark_min_time_seconds,
            args.calibration_random_data_size_per_device_mb,
            args.calibration_monotonic_access_stride,
            [d[0] for d in all_devices],
            args.calibration_num_concurrent_threads,
            args.calibration_num_reader_threads,
            args.calibration_datatype_modes,
            args.calibration_access_patterns,
        )

        calibration_results = {}
        for microbenchmark_results in res["benchmarks"]:
            run = microbenchmark_results["name"].split(";")
            logging.debug(f"run: {run}")

            assert (
                len(run) == 8
            ), f"expected benchmark name to be named TieringCalibration <access_pattern> <device> <num_tuples_scanned_per_iteration> <benchmark_time_multiplicator> <datatype> <bytes_per_value> <opts> but was {run}"
            access_pattern = run[1]
            device_name = hyrise_device_name_to_device_name(run[2])
            num_tuples_scanned_per_iteration = int(run[3])
            benchmark_time_multiplicator_to_divide_by = int(run[4])
            datatype = run[5]
            bytes_per_value = float(run[6])

            run_time = float(microbenchmark_results["real_time"])

            logging.debug(
                f"num_tuples_scanned_per_iteration: {num_tuples_scanned_per_iteration}"
            )
            logging.debug(
                f"benchmark_time_multiplicator_to_divide_by: {benchmark_time_multiplicator_to_divide_by}"
            )
            logging.debug(f"run_time: {run_time}")

            device_id = device_names_to_ids[device_name]

            key = f"{device_id}_{datatype}"
            if key not in calibration_results:
                calibration_results[key] = DeviceCalibrationResult(
                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, device_id, device_name, datatype
                )

            setattr(
                calibration_results[key],
                access_pattern + "_accesses_runtime",
                run_time
                * (1.0 / num_tuples_scanned_per_iteration)
                * (1.0 / benchmark_time_multiplicator_to_divide_by)
                * (1.0 / bytes_per_value),
            )

            logging.debug(calibration_results)

        for k, v in calibration_results.items():
            v.random_accesses_runtime = (
                v.random_single_chunk_accesses_runtime
                + v.random_multiple_chunk_accesses_runtime
            ) / 2.0

        return calibration_data_to_df(calibration_results)


def run_calibration(
    server_config: HyriseServerConfig,
    args,
) -> DeviceCalibrationResults:
    logging.info("Running Calibration")

    if args.run_calibration:
        logging.info("Measuring new calibration data")
        df = get_measured_calibration_data(args.all_devices, server_config, args)
    elif args.calibration_file:
        logging.info("Using existing calibration data")
        df = pd.read_csv(args.calibration_file)
    else:
        logging.info("Using artificial calibration data")
        df = get_static_calibration_data(args.all_devices, args)

    df.to_csv(CALIBRATION_FILE)

    calibration_data = calibration_df_to_dict(df)

    logging.info(
        f"Calibration finished. Results in {CALIBRATION_FILE}: {calibration_data}"
    )

    return calibration_data
