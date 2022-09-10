from functools import wraps
from time import time
from timeit import default_timer as timer
from typing import Any, Callable, Generic, NamedTuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


# Use for typing (cannot inherit from NamedTuple at the same time)
# keep in sync with TimedReturn
class TimedReturnT(Generic[T]):
    result: T
    runtime_milliseconds: float
    start_time_epoch_seconds: float
    end_time_epoch_seconds: float


# Use for instantiation
# keep in sync with TimedReturn
class TimedReturn(NamedTuple):
    result: Any
    runtime_milliseconds: float
    start_time_epoch_seconds: float
    end_time_epoch_seconds: float


def timed(f: Callable[..., U]) -> Callable[..., TimedReturnT[U]]:
    @wraps(f)
    def wrap(*args, **kw):
        start_time_epoch_s = time()
        start = timer()
        result = f(*args, **kw)
        end = timer()
        duration = end - start
        return TimedReturn(result, duration * 1000.0, start_time_epoch_s, time())

    return wrap
