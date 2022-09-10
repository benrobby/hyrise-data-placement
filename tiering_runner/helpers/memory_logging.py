import atexit
import datetime
import logging
import os
import shutil
import subprocess
import sys
import threading
from time import sleep, time
from typing import List

from tiering_runner.helpers.globals import SYSTEM_UTILIZATION_DIR
from tiering_runner.helpers.propagating_thread import PropagatingThread
from tiering_runner.hyrise_server.hyrise_interface import HyriseInterface

DEVICE_LOGGER_PROCS = []


def kill_proc(proc):
    if proc is not None:
        proc.kill()
        while proc.poll() is None:
            sleep(1)


def shut_down_logger_procs():
    global DEVICE_LOGGER_PROCS
    for proc in DEVICE_LOGGER_PROCS:
        kill_proc(proc)
    kill_proc(HYRISE_LOGGING_THREAD)


atexit.register(shut_down_logger_procs)

i = 0

HYRISE_LOGGING_THREAD = None
finish_running_events = []

SLEEP_TIME_SECONDS = 10


def set_up_new_server_loggers(server, pid):
    global finish_running_events

    for f in finish_running_events:
        f.set()
    finish_running_events = [threading.Event(), threading.Event()]

    start_hyrise_system_utilization(server, finish_running_events[0])
    start_hyrise_dram_pmap(pid)
    start_hyrise_jemalloc_all_devices(server, finish_running_events[1])


def start_hyrise_system_utilization(server, finish_running_event):
    def append_system_utilization(server, finish_event):
        try:

            file = SYSTEM_UTILIZATION_DIR / "hyrise_system_utilization.csv"
            file.touch()

            client = HyriseInterface(server)

            while not finish_event.is_set():
                sleep(SLEEP_TIME_SECONDS)
                df = client.get_meta_system_utilization(False)
                initialize_file = file.stat().st_size == 0
                df["timestamp"] = datetime.datetime.now()
                df.to_csv(file, mode="a", header=initialize_file, index=False)
        except Exception as e:
            logging.error(f"System utilization logger failed with {e}")
            raise e

    thread = PropagatingThread(
        target=append_system_utilization, args=(server, finish_running_event)
    )
    thread.daemon = True
    thread.start()


def start_hyrise_dram_pmap(hyrise_pid):
    global i
    global HYRISE_LOGGING_THREAD
    kill_proc(HYRISE_LOGGING_THREAD)
    device_command = (
        f"pmap -q -x {hyrise_pid} "
        + """| sed '1d' | gawk '{ print strftime("%Y-%m-%d_%H:%M:%S"), $0 }' | sed 's/\[ /\[/g' | sed 's/ \]/\]/g' | sed 's/  */,/g' | sed 's/,(deleted)//g'"""
    )
    print(device_command)
    device_name_escaped = f"pmap_DRAM_hyrise"
    HYRISE_LOGGING_THREAD = start_cmd_logging_thread(
        device_name_escaped,
        device_command,
        init=HYRISE_LOGGING_THREAD is None,
        header="timestamp,address,kbytes,rss,dirty,mode,mapping\n",
    )


df_counter = 0


def start_hyrise_jemalloc_all_devices(server, finish_running_event):
    return

    def append_jemalloc_stats(server, finish_event):
        logger = logging.getLogger("jemalloc_stats_logger")
        logger.info("Jemalloc stats logger starting")
        try:
            global df_counter
            file = SYSTEM_UTILIZATION_DIR / "hyrise_jemalloc_all_devices.csv"
            file.touch()

            hyrise_client = HyriseInterface(server)

            while not finish_event.is_set():
                sleep(SLEEP_TIME_SECONDS)
                df = hyrise_client.get_tiering_device_information()
                df["index"] = df_counter
                df["timestamp"] = datetime.datetime.now()
                initialize_file = file.stat().st_size == 0
                if df.empty:
                    raise Exception(
                        "Empty response from server for jemalloc device utilization logging"
                    )

                df.to_csv(file, mode="a", header=initialize_file, index=False)
                df_counter += 1
        except Exception as e:
            logger.error(f"Jemalloc stats logger failed with {e}")
            raise e

    thread = PropagatingThread(
        target=append_jemalloc_stats, args=(server, finish_running_event)
    )
    thread.daemon = True
    thread.start()


def start_cmd_logging_thread(
    device_name_escaped,
    device_command,
    init=True,
    header="consumption,device_name\n",
):
    device_file = f"{SYSTEM_UTILIZATION_DIR}/{device_name_escaped}.csv"
    if init:
        with open(device_file, "a") as f:
            f.write(header)

    looping_command = f"while true\ndo\n  {device_command} >> {device_file}\n  sleep {SLEEP_TIME_SECONDS}\ndone"
    proc = subprocess.Popen(
        [looping_command],
        shell=True,
    )
    return proc


def set_up_file_device_logging(device_name):
    device_name_escaped = "umap_filesize_" + device_name.replace("/", "_")
    device_command = f"du -s --block-size=1 {device_name} | sed 's/\t/,/g'"
    # if file at device_name exists
    if os.path.exists(device_name):
        DEVICE_LOGGER_PROCS.append(
            start_cmd_logging_thread(device_name_escaped, device_command)
        )
