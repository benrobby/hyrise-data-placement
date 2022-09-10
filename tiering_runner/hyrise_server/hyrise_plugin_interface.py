import json
import logging
import sys
import warnings
from time import time
from typing import Any, Dict, List

from tiering_runner.hyrise_server.utils import query_to_one_line

warnings.simplefilter(action="ignore", category=UserWarning)

import pandas as pd
import psycopg2
from tiering_runner.helpers.globals import (BENCHMARK_JOB, BENCHMARKS_DIR,
                                            PLUGIN_FILETYPE, TMP_DIR,
                                            device_name_to_hyrise_device_name)

from tiering_runner.hyrise_server.hyrise_server import HyriseServer

PLUGIN_NAMES = ["TieringSelectionPlugin", "WorkloadHandlerPlugin"]

logger = logging.getLogger("hyrise_plugin_client")
logger.setLevel(logging.DEBUG)


class HyrisePluginInterface:
    def __init__(
        self,
        server: HyriseServer,
    ) -> None:
        self.config = server.config
        self.server = server
        self.execute_lock = self.server.execute_lock

        self.connection = psycopg2.connect(
            f"host=localhost port={self.config.port}",
            options="-c statement_timeout=30000",
        )
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.ensure_server_is_initialized()

    def ensure_server_is_initialized(self):
        if not self.server.initialized:
            self.server.initialized = True
            self._load_plugins()
            if self.config.benchmark_name == BENCHMARK_JOB:
                self._load_job_data()
            self.server.config.running_server_is_initialized = True
            self._set_num_cores()

    def _format(self, command: str, *format_args):
        return command.format(*[str(arg) for arg in list(format_args)])

    def _send_sql(self, command: str, *format_args):
        cmd = self._format(command, *format_args)

        with self.execute_lock.acquire_timeout(30) as result:
            if not result:
                raise Exception("Timeout when acquiring execute lock")
            logger.debug("Execute mutex is acquired")
            logger.info(f"sending sql: {query_to_one_line(cmd)}")
            self.cursor.execute(cmd)
            result = None
            try:
                result = self.cursor.fetchall()
            except Exception as e:
                logging.debug(f"Error fetching result: {e}")

            col_names = {}
            if self.cursor.description is not None:
                for item_id, item in enumerate(self.cursor.description):
                    col_names[item.name] = item_id

        logger.debug("Execute mutex is released")

        return result, col_names

    def _plugin_call(self, command: str):

        with self.execute_lock.acquire_timeout(30) as result:
            if not result:
                raise Exception("Timeout when acquiring execute lock")
            logger.debug("Execute mutex is acquired")
            cmd = self._format(
                """UPDATE meta_settings
                        SET value='{}'
                        WHERE name='Plugin::Tiering::Command';""",
                command,
            )
            logger.info(f"sending plugin call: {query_to_one_line(cmd)}")

            self.cursor.execute(cmd)

            self.cursor.execute("SELECT * FROM meta_tiering_command;")
            result = self.cursor.fetchall()
        logger.debug("Execute mutex is released")

        return result

    def _send_plugin_command_sync(self, command: str) -> List[str]:

        result = self._plugin_call(command)

        assert (
            len(result) >= 1 and len(result) <= 2
        ), f"Error sending command: {command}: {result}"
        if not any(res[0] == "Command executed successfully." for res in result):
            logger.error(f"Error sending command: {command}: {result}")
            sys.exit(1)

        return_values = [
            r[0] for r in result if r[0] != "Command executed successfully."
        ]
        if len(return_values) > 0:
            logger.info(f"got return: {return_values}")

        return return_values

    def _load_plugins(self):
        for plugin_name in PLUGIN_NAMES:
            self._send_sql(
                """INSERT INTO meta_plugins(name)
                VALUES ('{}/lib{}{}');""",
                self.config.hyrise_server_executable_path,
                plugin_name,
                PLUGIN_FILETYPE,
            )

    def _load_job_data(self):

        data_path = str(self.config.job_data_path)
        logging.info(
            'Loading "Join Order Benchmark" data',
        )

        def load_table_bin(name):
            try:
                logging.info(f"Loading JOB table {name}")
                self._send_sql(f"COPY {name} from '{data_path}/{name}.bin';")
            except Exception as e:
                sys.exit(f"Error when loading JOB table {name} {data_path}: {e}")

        start_time = time()
        job_tables = [
            "title",
            "kind_type",
            "role_type",
            "link_type",
            "keyword",
            "company_type",
            "movie_link",
            "complete_cast",
            "person_info",
            "comp_cast_type",
            "aka_title",
            "movie_info",
            "company_name",
            "movie_keyword",
            "movie_companies",
            "cast_info",
            "info_type",
            "aka_name",
            "movie_info_idx",
            "char_name",
            "name",
        ]
        for t in job_tables:
            load_table_bin(t)

        logging.info(f"Loading JOB data done in {time() - start_time} seconds")

    def _set_num_cores(self):
        self._send_plugin_command_sync(f"SET SERVER CORES {self.config.cores}")

    def run_calibration(
        self,
        scale_factor: float,
        benchmark_min_time_seconds: float,
        random_data_size_per_device_mb: int,
        monotonic_access_stride: int,
        devices: List[str],
        num_concurrent_threads: int,
        num_reader_threads: int,
        modes: str,
        access_patterns: str,
    ) -> Dict[str, Any]:

        filename = (TMP_DIR / "calibration_measurements.json").expanduser().resolve()
        with open(filename, "w") as f:
            f.write("")

        devices = [d for d in devices if d != 'UNUSED']

        devices_str = " ".join([device_name_to_hyrise_device_name(d) for d in devices])
        self._send_plugin_command_sync(
            f"RUN TIERING CALIBRATION {filename} {scale_factor} {benchmark_min_time_seconds} {random_data_size_per_device_mb} {monotonic_access_stride} {num_concurrent_threads} {num_reader_threads} {modes} {access_patterns} {devices_str}"
        )

        with open(filename, "r") as f:
            return json.load(f)

    def apply_tiering_configuration(self, tiering_config: pd.DataFrame):
        def write_config_to_table_major_json(config: pd.DataFrame):
            df = config.copy()
            group = df.groupby("table_name").apply(
                lambda x: x[
                    ["column_id", "chunk_id", "device_id", "device_name"]
                ].to_dict(orient="records"),
            )
            d = {"configuration": dict(group)}
            filename = (TMP_DIR / "tiering_config.json").expanduser().resolve()
            with open(filename, "w") as f:
                json.dump(d, f, indent=4)
            return filename

        filename = write_config_to_table_major_json(tiering_config)
        self._send_plugin_command_sync(f"APPLY TIERING CONFIGURATION {filename}")

    def set_devices(self, umap_buf_size_bytes: int, devices: List[str]):
        self._send_plugin_command_sync(
            f"SET DEVICES {umap_buf_size_bytes} {' '.join(devices)}"
        )

    def visualize_query(self, query: str, query_name: str, test_id: str):
        filename = (TMP_DIR / "query.sql").expanduser().resolve()
        with open(filename, "w") as f:
            f.write(query)

        self._send_plugin_command_sync(
            f"VIS_QUERY;{query_name};{test_id};{filename};{str(BENCHMARKS_DIR.absolute())}"
        )

    def clear_umap_buffer(self):
        self._send_plugin_command_sync("CLEAR_UMAP_BUFFER")
