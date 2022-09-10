from tiering_runner.helpers.types import TieringAlgorithmInput, TieringAlgorithmResult


class TieringAlgorithm:
    def __init__(self, input: TieringAlgorithmInput):
        self.input = input

    def set_up_solver(self) -> None:
        raise NotImplementedError()

    def solve(self) -> None:
        raise NotImplementedError()

    def get_solver_result(self) -> TieringAlgorithmResult:
        raise NotImplementedError()
