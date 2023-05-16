from collections import deque
from typing import Iterable, Optional

INPUTS: deque[str] = deque()

original_input = input


def input(prompt: str) -> str:
    try:
        cached_input = INPUTS.popleft()
        print(prompt, cached_input)
        return cached_input
    except IndexError:
        return original_input(prompt)


def next_inputs(to_add: Optional[Iterable[str]] = None) -> deque[str]:
    if to_add is not None:
        INPUTS.extend(to_add)
    return INPUTS


def clear_next_inputs() -> None:
    INPUTS.clear()
