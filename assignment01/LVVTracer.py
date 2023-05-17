import sys
from collections import defaultdict
from copy import deepcopy
from types import FrameType
from typing import Any, Callable, Mapping, Optional

TraceFunction = Callable[[FrameType, str, Any], Optional["TraceFunction"]]


class LVVTracer:
    def __init__(self, target_func: str):
        self.target_func = target_func
        self.local_values: defaultdict[str, Any] = defaultdict(object)
        self.local_changes: defaultdict[str, int] = defaultdict(int)

    def __enter__(self) -> "LVVTracer":
        self.original_trace_func = sys.gettrace()
        sys.settrace(self.global_tracer)
        return self

    def __exit__(self, *_: Any) -> bool:
        sys.settrace(self.original_trace_func)

        # NOTE: Errors are always propagated, the Debugging Book returns None or False:
        #       which are both false-y and result in errors being propagated.
        return False

    def do_count(self, locals: Mapping[str, Any]) -> None:
        for name, current in locals.items():
            if self.local_values[name] != current:
                self.local_values[name] = deepcopy(current)
                self.local_changes[name] += 1

    def global_tracer(self, frame: FrameType, event: str, arg: Any) -> Optional[TraceFunction]:
        if event == "call" and frame.f_code.co_name == self.target_func:
            self.do_count(frame.f_locals)
            return self.target_tracer

    def target_tracer(self, frame: FrameType, event: str, arg: Any) -> Optional[TraceFunction]:
        self.do_count(frame.f_locals)

        # Fulfill requirement 9
        if event == "return":
            self.local_values.clear()

        return self.target_tracer

    def getLVVmap(self) -> dict[str, int]:
        return self.local_changes
