import pandas as pd
from tiering_runner.determine_tiering.tiering_algorithm import TieringAlgorithm
from tiering_runner.helpers.globals import (
    DRAM_DEVICE,
    OBJECTIVE_MODE_DEVICE_BUDGET,
    SEGMENT_ACCESS_PATTERNS,
)
from tiering_runner.helpers.timing import timed
from tiering_runner.helpers.types import (
    SegmentTieringAssignment,
    TieringAlgorithmInput,
    TieringAlgorithmMappings,
    TieringAlgorithmResult,
)


class DetermineTieringDram(TieringAlgorithm):
    def __init__(self, input: TieringAlgorithmInput):
        super().__init__(input)

    def set_up_solver(self):
        return

    @timed
    def _solve(self):
        input = self.input
        tiering_config = []
        mappings = input.mappings
        assert (
            input.objective_mode == OBJECTIVE_MODE_DEVICE_BUDGET
        ), f"Only {OBJECTIVE_MODE_DEVICE_BUDGET} is supported"

        for segment_index, segment in input.segments_df.iterrows():
            table_id = int(segment["table_id"])
            column_id = int(segment["column_id"])
            chunk_id = int(segment["chunk_id"])
            segment_size_bytes = int(segment["size_in_bytes"])
            device_id = mappings.device_names_to_ids[DRAM_DEVICE]
            assignment = SegmentTieringAssignment(
                table_id,
                column_id,
                chunk_id,
                device_id,
                mappings.table_ids_to_names[table_id],
                mappings.device_ids_to_names[device_id],
                segment_size_bytes,
                sum(int(segment[p]) for p in SEGMENT_ACCESS_PATTERNS),
            )
            tiering_config.append(assignment)

        return pd.DataFrame(tiering_config)

    def solve(self):
        res = self._solve()
        self.tiering_df = res.result
        self.runtime_ms = res.runtime_milliseconds

    def get_solver_result(self):
        return (
            self.tiering_df,
            self.input.mappings.device_ids_to_names,
            self.input.mappings.table_ids_to_names,
            0,
            True,
            self.runtime_ms,
        )
