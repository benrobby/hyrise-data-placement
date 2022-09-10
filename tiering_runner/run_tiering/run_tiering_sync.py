import logging
import traceback
from timeit import default_timer as timer
from typing import List, Tuple

import pandas as pd
from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.determine_tiering.determine_tiering import (
    determine_tiering_for_spec,
)
from tiering_runner.helpers.types import (
    BenchmarkConfig,
    DeviceCalibrationResults,
    TieringConfigMetadata,
)
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface
from tiering_runner.hyrise_server.hyrise_plugin_interface import HyrisePluginInterface
from tiering_runner.hyrise_server.hyrise_server import (
    HYRISE_SERVER_PROCESS,
    HyriseServer,
)

CURRENT_TIERING_CONFIG: Tuple[pd.DataFrame, TieringConfigMetadata] = None


def run_tiering_sync(
    server: HyriseServer,
    benchmark_config: BenchmarkConfig,
    spec: TieringBenchmarkSpec,
    calibration_data: DeviceCalibrationResults,
    args,
    meta_segments: pd.DataFrame,
    cache_config=False,
):
    global CURRENT_TIERING_CONFIG

    config_metadata = TieringConfigMetadata(
        spec.benchmark_name,
        spec.m_scale_factor(),
        spec.optimization_method,
        spec.m_devices(),
    )

    if CURRENT_TIERING_CONFIG is not None:
        if CURRENT_TIERING_CONFIG[1] == config_metadata and cache_config:
            logging.info(
                f"Reusing Tiering Configuration: Server already has this tiering configuration {config_metadata}"
            )
            return

    try:
        conf = determine_tiering_for_spec(
            server, benchmark_config, spec, meta_segments, calibration_data, args
        )

        if args.tiering_apply_configuration:
            plugin_client = HyrisePluginInterface(server)
            plugin_client.apply_tiering_configuration(conf)

        CURRENT_TIERING_CONFIG = (conf, config_metadata)
    except Exception as e:
        if not args.tiering_apply_configuration:
            logging.warn(e)
            traceback.print_exc()
            logging.warn(
                f"Determine Tiering Failed. Skipping this config because we do not need to apply it. {spec.dollar_budget_cents}"
            )
        else:
            raise e
