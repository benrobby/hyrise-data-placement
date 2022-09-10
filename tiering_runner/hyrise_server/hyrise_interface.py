import logging
from typing import Dict

import pandas as pd
import pandas.io.sql as psql
import psycopg2
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import QueryResultT
from tiering_runner.hyrise_server.hyrise_server import HyriseServer
from tiering_runner.hyrise_server.utils import query_to_one_line

logger = logging.getLogger("hyrise_client")
logger.setLevel(logging.DEBUG)

# thin wrapper around psycopg2 cursor
# thread-safe, multiple threads should instantiate their own HyriseInterface
# Blocking within the same thread, but multiple threads can wait on the results
# of multiple queries at the same time.
class HyriseInterface:
    def __init__(self, server: HyriseServer) -> None:
        self.config = server.config
        self.connection = psycopg2.connect(
            f"host=localhost port={self.config.port}",
            options="-c statement_timeout=30000",
        )
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()

    def _format(self, command: str, *format_args):
        return command.format(*[str(arg) for arg in list(format_args)])

    def _send_sql_and_get_result_df(self, query, args=None, should_log=True) -> QueryResultT:
        q = self._format(query, args)
        if should_log:
            logger.debug(f"Sending Hyrise Query: {query_to_one_line(q)}")
        df = psql.read_sql(q, self.connection)
        csv = df.to_csv().replace("\n", ";")

        if should_log:
            logger.debug(f"Got response to query {query_to_one_line(q)} : {csv}")

        return df

    def get_meta_segments(self, should_log=True) -> pd.DataFrame:
        df = self._send_sql_and_get_result_df(
            """SELECT *
            FROM meta_segments;""",  # todo use accurate?,
            should_log=should_log,
        )
        df["size_in_bytes"] = df["estimated_size_in_bytes"]
        return df

    def get_meta_system_utilization(self, should_log=True):
        return self._send_sql_and_get_result_df(
            """SELECT *
            FROM meta_system_utilization;""",
            should_log=should_log
        )

    def get_tiering_device_information(self, should_log=True):
        return self._send_sql_and_get_result_df(
            """SELECT *
            FROM meta_tiering_device_information;""",
            should_log=should_log
        )

    def get_benchmark_queries(self) -> Dict[str, str]:

        queries_df = self._send_sql_and_get_result_df(
            """SELECT *
            FROM meta_benchmark_items
            WHERE benchmark_name = '{}';""",
            self.config.benchmark_name[1],
        )

        assert queries_df.shape[0] > 0, "no queries found for benchmark"

        benchmark_queries = {}

        for row_index, row in queries_df.iterrows():
            k = row["item_name"]
            if k not in benchmark_queries:
                benchmark_queries[k] = []

            benchmark_queries[k].append(row["sql_statement_string"])

        return benchmark_queries

    @timed
    def run_benchmark_query(self, query: str, query_name: str) -> None:
        logger.debug(f"Sending Benchmark Query {query_name}")
        # logger.debug(f"Query is: {query_to_one_line(query)}")

        self.cursor.execute(query)
        self.cursor.fetchall()
