import logging
from copy import deepcopy
from typing import Tuple

from tiering_runner.config_generation.json_benchmark_config import TieringBenchmarkSpec
from tiering_runner.helpers.globals import to_long_name
from tiering_runner.helpers.memory_logging import set_up_new_server_loggers
from tiering_runner.helpers.types import HyriseServerConfig
from tiering_runner.hyrise_server.hyrise_plugin_interface import HyrisePluginInterface
from tiering_runner.hyrise_server.hyrise_server import (
    HYRISE_SERVER_PROCESS,
    HyriseServer,
)


def create_new_server(old_server, new_server_config, args):
    if old_server is not None:
        old_server.stop()
    logging.info("Not reusing existing server, creating new server")
    server = HyriseServer(new_server_config)
    pid = server.start()

    set_up_new_server_loggers(server, pid)

    logging.info("Server was newly created, setting up plugins etc.")
    plugin_client = HyrisePluginInterface(server)
    plugin_client.ensure_server_is_initialized()  # early on ensure that plugin loading works
    plugin_client.set_devices(args.umap_buf_size_bytes, args.all_device_names)

    return server, True


def should_create_new_server(
    force_create_new_server, args, old_server, new_server_config
):
    if force_create_new_server:
        return True
    if args.run_benchmarks_always_restart_server:
        return True
    if old_server is None:
        return True

    if (
        old_server.config.benchmark_name == new_server_config.benchmark_name
        and old_server.config.scale_factor == new_server_config.scale_factor
        and old_server.config.cores == new_server_config.cores
        and old_server.config.encoding == new_server_config.encoding
    ):
        return False

    return True


def create_or_reuse_server(
    old_server: HyriseServer,
    server_config: HyriseServerConfig,
    spec: TieringBenchmarkSpec,
    args,
    force_create_new_server=False,
) -> Tuple[HyriseServer, bool]:

    benchmark_name = (
        spec.benchmark_name,
        to_long_name(spec.benchmark_name),
    )

    new_server_config = deepcopy(server_config)
    new_server_config.benchmark_name = benchmark_name
    new_server_config.scale_factor = spec.m_scale_factor()
    new_server_config.cores = spec.num_cores
    new_server_config.encoding = spec.encoding

    should_create_new = should_create_new_server(
        force_create_new_server, args, old_server, new_server_config
    )

    if should_create_new:
        return create_new_server(old_server, new_server_config, args)
    else:
        logging.info(
            f"Reusing server for {new_server_config.benchmark_name} {spec.m_scale_factor()} {spec.num_cores} {spec.encoding}"
        )
        return old_server, False
