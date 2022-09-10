import atexit
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime

from tiering_runner.helpers.globals import HYRISE_SERVER_RUN_LOG_FILE
from tiering_runner.helpers.types import HyriseServerConfig

HYRISE_SERVER_PROCESS = None

logger = logging.getLogger("hyrise_server")


# global so we can call exit handler
def shut_down_server():
    global HYRISE_SERVER_PROCESS
    if HYRISE_SERVER_PROCESS is not None and HYRISE_SERVER_PROCESS.poll() is None:
        logger.info("Shutting down Hyrise Server...")
        HYRISE_SERVER_PROCESS.kill()
        wait_seconds = 0
        while HYRISE_SERVER_PROCESS.poll() is None:
            time.sleep(1)
            wait_seconds += 1
            if wait_seconds > 60 * 5:
                logger.info("Calling terminate as hyrise Server doesn't want to stop")
                HYRISE_SERVER_PROCESS.terminate()
        logger.info("Shutdown successful.")


atexit.register(shut_down_server)


class HyriseServer:
    def __init__(self, config: HyriseServerConfig) -> None:
        self.config = config
        self._is_running = config.attach_to_running_server
        self.initialized = (
            config.attach_to_running_server and config.running_server_is_initialized
        )
        self.execute_lock = threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _set_up_logger_thread(self, server_has_started_condition):
        def log_and_check_server_start(pipe):
            with open(
                HYRISE_SERVER_RUN_LOG_FILE,
                "a",
                buffering=1,
                encoding="utf-8",
            ) as log_file:

                def write_line(line: str):
                    log_file.write(f"{datetime.now()}:\t{line}")
                    sys.stdout.flush()

                exit_on_server_stop = False
                for line in iter(pipe.readline, ""):
                    if line:
                        if "Server started at" in line:
                            server_has_started_condition.set()
                        write_line(line)
                        if "terminate called" in line:
                            logger.error("Error: exception in server, stopping.")
                            exit_on_server_stop = True

                write_line("=== Server stopped.\n")
                pipe.close()
                if exit_on_server_stop:
                    sys.exit("Exiting due to server exception.")

        logger_thread = threading.Thread(
            target=log_and_check_server_start, args=(HYRISE_SERVER_PROCESS.stdout,)
        )
        logger_thread.daemon = True
        logger_thread.start()

    def _wait_until_server_has_started(self, server_has_started_condition):
        logger.info("Waiting for Hyrise to start: ")
        server_start_time = time.time()
        while not server_has_started_condition.wait(timeout=5):
            logger.info(".")
            if (
                time.time() - min(1200, 120000 * self.config.scale_factor)
                > server_start_time
            ):
                logger.error("Error: time out during server start")
                return
        logger.info(f"HyriseServer {self.config} started successfully.")

    def _get_benchmark_data_string(self):
        if self.config.benchmark_name is None:
            return ""

        name_short = self.config.benchmark_name[0]

        if name_short == "JOB":
            return ""  # will be loaded from python

        additional_options = ":skewed" if name_short == "JCCH" else ""

        return f"--benchmark_data={self.config.benchmark_name[0].lower()}:{self.config.scale_factor}:{self.config.encoding}{additional_options}"

    def start(self):
        if self._is_running:
            logger.info("HyriseServer is already running, not restarting.")
            return

        global HYRISE_SERVER_PROCESS

        logger.info(
            f"Starting Hyrise server (not NUMA-bound) with config {self.config}) ... "
        )

        benchmark_data_string = self._get_benchmark_data_string()

        call = [
            "{}/hyriseServer".format(str(self.config.hyrise_server_executable_path)),
            "-p",
            str(self.config.port),
            benchmark_data_string,
        ]
        logger.info(f"calling hyriseServer with: {str(call)}")
        HYRISE_SERVER_PROCESS = subprocess.Popen(
            call,
            cwd=self.config.hyrise_dir,
            stdout=subprocess.PIPE,
            bufsize=0,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        server_has_started_condition = threading.Event()

        self._set_up_logger_thread(server_has_started_condition)
        self._wait_until_server_has_started(server_has_started_condition)

        self._is_running = True

        return HYRISE_SERVER_PROCESS.pid

    def stop(self):
        shut_down_server()
        self._is_running = False
