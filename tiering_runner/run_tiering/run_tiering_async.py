import logging
import threading
from datetime import datetime
from time import sleep, time
from timeit import default_timer as timer
from typing import List, Tuple

import pandas as pd
import psycopg2
from tiering_runner.determine_tiering.determine_tiering import \
    determine_tiering
from tiering_runner.helpers.globals import (OBJECTIVE_MODE_DEVICE_BUDGET,
                                            TIERING_CONFIG_DIR,
                                            TIERING_RUNS_FILE)
from tiering_runner.helpers.propagating_thread import PropagatingThread
from tiering_runner.helpers.random_string import get_random_string
from tiering_runner.helpers.types import (DetermineTieringRunMetadata,
                                          DeviceCalibrationResults,
                                          TieringDevice)
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface
from tiering_runner.hyrise_server.hyrise_plugin_interface import \
    HyrisePluginInterface
from tiering_runner.hyrise_server.hyrise_server import (HYRISE_SERVER_PROCESS,
                                                        HyriseServer)
from tiering_runner.run_tiering.get_meta_segments import (
    get_meta_segments, get_meta_segments_raw)
from tiering_runner.run_tiering.get_meta_segments_delta import \
    get_meta_segments_delta

CURRENT_TIERING_THREAD = None
CURRENT_FINISH_EVENT = None

logger = logging.getLogger("run_tiering_async")


def sleep_periodically_check_finish_event(sleep_time, finish_event):
    num_checks = 20
    for i in range(num_checks):
        sleep(sleep_time / num_checks)
        if finish_event.is_set():
            return False
    return True


def async_tiering_worker(
    server: HyriseServer,
    args,
    calibration_data: DeviceCalibrationResults,
    optimization_method: str,
    cost_model_name: str,
    devices: List[TieringDevice],
    benchmark_queries,
    finish_event: threading.Event,
):

    worker_id = get_random_string()

    def get_run_id(run: int):
        return f"async_tiering_worker_{worker_id}_{run}"

    worker_logger = logging.getLogger(f"async_tiering_worker_{worker_id}")
    worker_logger.info(
        f'Starting async tiering worker "{worker_id}" for {optimization_method} {devices}'
    )

    try:
        run = 0
        old_meta_segments = get_meta_segments_raw(
            args, server, "-1_async_tiering"
        )

        while not finish_event.is_set():
            should_continue = sleep_periodically_check_finish_event(
                args.tiering_timeseries_config_update_wait_time_s, finish_event
            )
            if not should_continue:
                break

            run_id = get_run_id(run)
            worker_logger.info(f"Applying new tiering in {run_id}")
            with open(TIERING_RUNS_FILE, "a") as f:
                f.write(
                    f'{optimization_method},{run_id},{run},{datetime.now().isoformat()},"{devices}"\n'
                )

            current_meta_segments = get_meta_segments_raw(
                args,
                server,
                f"{run}_async_tiering",
            )
            meta_segments_delta = get_meta_segments_delta(
                current_meta_segments, old_meta_segments
            )
            old_meta_segments = current_meta_segments

            meta_segments_delta.to_csv(
                TIERING_CONFIG_DIR / f"{run}_async_tiering_meta_segments_delta.csv",
                sep=",",
            )

            run_metadata = DetermineTieringRunMetadata(
                server.config.benchmark_name[0],
                server.config.scale_factor,
                str(run),
                str(run),
                f"{run}_async_tiering",
            )

            tiering_config = determine_tiering(
                server,
                None,
                run_metadata,
                meta_segments_delta,
                calibration_data,
                optimization_method,
                devices,
                args,
                cost_model_name,
                OBJECTIVE_MODE_DEVICE_BUDGET,
                -1,
                -1,
            )

            if args.tiering_apply_configuration:
                try:
                    plugin_client = HyrisePluginInterface(server)
                    plugin_client.apply_tiering_configuration(tiering_config)
                except psycopg2.OperationalError as e:
                    if finish_event.is_set():
                        logger.info(
                            f"Tiering runner stopping, server was already shut down: {e}"
                        )
                        break
                    else:
                        raise e

            run += 1

        worker_logger.info(f"Finished async tiering worker after {run} iterations")

    except Exception as e:
        worker_logger.exception(f"Exception in async tiering worker: {e}")
        raise e


def start_tiering_worker_async(
    server: HyriseServer,
    calibration_data: DeviceCalibrationResults,
    args,
    optimization_method: str,
    cost_model_name: str,
    devices: List[TieringDevice],
    benchmark_queries,
):
    global CURRENT_TIERING_THREAD
    global CURRENT_FINISH_EVENT

    # start tiering worker thread (set finish event, join old thread, start new thread)
    # tiering runner params: wait time, finish event, server, optimization_method

    if CURRENT_TIERING_THREAD is not None:
        logger.info("Stopping old tiering thread")
        if CURRENT_FINISH_EVENT is not None:
            CURRENT_FINISH_EVENT.set()
        CURRENT_TIERING_THREAD.join()
        logger.info("Old tiering thread stopped")

    CURRENT_FINISH_EVENT = threading.Event()

    logger.info("Starting new tiering thread")
    CURRENT_TIERING_THREAD = PropagatingThread(
        target=async_tiering_worker,
        args=(
            server,
            args,
            calibration_data,
            optimization_method,
            cost_model_name,
            devices,
            benchmark_queries,
            CURRENT_FINISH_EVENT,
        ),
    )
    CURRENT_TIERING_THREAD.daemon = True
    CURRENT_TIERING_THREAD.start()
