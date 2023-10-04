import sys
from collections import defaultdict
from copy import deepcopy
from types import FrameType
from typing import Any, Callable, Mapping, Optional, Iterable

TraceFunction = Callable[[FrameType, str, Any], Optional["TraceFunction"]]


class LFTracer:
    def __init__(self, target_func: Iterable[str]):
        self.target_func = set(target_func)
        self.line_hits: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))

    def __enter__(self) -> "LFTracer":
        self.original_trace_func = sys.gettrace()
        sys.settrace(self.global_tracer)
        return self

    def __exit__(self, *_: Any) -> bool:
        sys.settrace(self.original_trace_func)
        return False

    def do_count(self, frame: FrameType) -> None:
        if frame.f_code.co_name in self.target_func:
            self.line_hits[frame.f_code.co_name][frame.f_lineno] += 1

    def global_tracer(self, frame: FrameType, event: str, arg: Any) -> Optional[TraceFunction]:
        if event == "call" and frame.f_code.co_name in self.target_func:
            self.do_count(frame)
            return self.target_tracer

    def target_tracer(self, frame: FrameType, event: str, arg: Any) -> Optional[TraceFunction]:
        self.do_count(frame)
        return self.target_tracer

    def getLFMap(self) -> Mapping[str, Mapping[int, int]]:
        return self.line_hits
