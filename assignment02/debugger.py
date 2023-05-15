import inspect
import sys
import traceback
import warnings
from types import FrameType, FunctionType, TracebackType
from typing import Any, Callable, Optional, TextIO, Type, TypeVar

AnyCallable = Callable[..., Any]
BaseExceptionT = TypeVar("BaseExceptionT", bound=BaseException)
Location = tuple[AnyCallable, int]


class StackInspector:
    """Provide functions to inspect the stack"""

    def caller_frame(self) -> FrameType:
        """Return the frame of the caller."""

        # Walk up the call tree until we leave the current class
        frame = inspect.currentframe()
        assert frame

        while self.our_frame(frame):
            frame = frame.f_back
            assert frame

        return frame

    def our_frame(self, frame: FrameType) -> bool:
        """Return true if `frame` is in the current (inspecting) class."""
        return isinstance(frame.f_locals.get("self"), self.__class__)

    def caller_globals(self) -> dict[str, Any]:
        """Return the globals() environment of the caller."""
        return self.caller_frame().f_globals

    def caller_locals(self) -> dict[str, Any]:
        """Return the locals() environment of the caller."""
        return self.caller_frame().f_locals

    def caller_location(self) -> Location:
        """Return the location (func, lineno) of the caller."""
        return self.caller_function(), self.caller_frame().f_lineno

    def search_frame(
        self,
        name: str,
        frame: Optional[FrameType] = None,
    ) -> tuple[Optional[FrameType], Optional[AnyCallable]]:
        """
        Return a pair (`frame`, `item`)
        in which the function `name` is defined as `item`.
        """
        if frame is None:
            frame = self.caller_frame()

        while frame:
            item = None
            if name in frame.f_globals:
                item = frame.f_globals[name]
            if name in frame.f_locals:
                item = frame.f_locals[name]
            if item and callable(item):
                return frame, item

            frame = frame.f_back

        return None, None

    def search_func(self, name: str, frame: Optional[FrameType] = None) -> Optional[AnyCallable]:
        """Search in callers for a definition of the function `name`"""
        frame, func = self.search_frame(name, frame)
        return func

    # Avoid generating functions more than once
    _generated_function_cache: dict[tuple[str, int], AnyCallable] = {}

    def create_function(self, frame: FrameType) -> AnyCallable:
        """Create function for given frame"""
        name = frame.f_code.co_name
        cache_key = (name, frame.f_lineno)
        if cache_key in self._generated_function_cache:
            return self._generated_function_cache[cache_key]

        try:
            # Create new function from given code
            generated_function = FunctionType(frame.f_code, globals=frame.f_globals, name=name)
        except TypeError:
            # Unsuitable code for creating a function
            # Last resort: Return some function
            generated_function = self.unknown

        except Exception as exc:
            # Any other exception
            warnings.warn(
                f"Couldn't create function for {name} ({type(exc).__name__}: {exc})"
            )
            generated_function = self.unknown

        self._generated_function_cache[cache_key] = generated_function
        return generated_function

    def caller_function(self) -> AnyCallable:
        """Return the calling function"""
        frame = self.caller_frame()
        name = frame.f_code.co_name
        func = self.search_func(name)
        if func:
            return func

        if not name.startswith("<"):
            warnings.warn(f"Couldn't find {name} in caller")

        return self.create_function(frame)

    def unknown(self) -> None:  # Placeholder for unknown functions
        pass

    def is_internal_error(
        self,
        exc_tp: Type[BaseExceptionT],
        exc_value: BaseExceptionT,
        exc_traceback: TracebackType,
    ) -> bool:
        """Return True if exception was raised from `StackInspector` or a subclass."""
        if not exc_tp:
            return False

        for frame, _ in traceback.walk_tb(exc_traceback):
            if self.our_frame(frame):
                return True

        return False


class Tracer(StackInspector):
    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Trace a block of code, sending logs to `file` (default: stdout)"""
        self.original_trace_function: Optional[AnyCallable] = None
        self.file = file
        self.last_vars: dict[str, Any] = {}

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.print_debugger_status(frame, event, arg)

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[AnyCallable]:
        """Internal tracing function."""
        if self.our_frame(frame):
            # Do not trace our own methods
            pass
        else:
            self.traceit(frame, event, arg)
        return self._traceit

    def log(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = True) -> None:
        """
        Like `print()`, but always sending to `file` given at initialization,
        and flushing by default.
        """
        print(*objects, sep=sep, end=end, file=self.file, flush=flush)

    def __enter__(self) -> Any:
        """Called at begin of `with` block. Turn tracing on."""
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)

        # This extra line also enables tracing for the current block
        # inspect.currentframe().f_back.f_trace = self._traceit
        return self

    def __exit__(
        self,
        exc_tp: Type[BaseExceptionT],
        exc_value: BaseExceptionT,
        exc_traceback: TracebackType,
    ) -> bool:
        """
        Called at end of `with` block. Turn tracing off.
        Return `None` if ok, not `None` if internal error.
        """
        sys.settrace(self.original_trace_function)
        return False

    def changed_vars(self, new_vars: dict[str, Any]) -> dict[str, Any]:
        """Track changed variables, based on `new_vars` observed."""
        changed: dict[str, Any] = {}
        for var_name, var_value in new_vars.items():
            if var_name not in self.last_vars or self.last_vars[var_name] != var_value:
                changed[var_name] = var_value
        self.last_vars = new_vars.copy()
        return changed

    def print_debugger_status(self, frame: FrameType, event: str, arg: Any) -> None:
        """Show current source line and changed vars"""
        if event == "exception":
            exception, value, _ = arg
            self.log(f"{frame.f_code.co_name}() " f"raises {exception.__name__}: {value}")
            return

        changes = self.changed_vars(frame.f_locals)
        changes_s = ", ".join([var + " = " + repr(changes[var]) for var in changes])

        if event == "call":
            self.log("Calling " + frame.f_code.co_name + "(" + changes_s + ")")
        elif changes:
            self.log(" " * 40, "#", changes_s)

        if event == "line":
            try:
                module = inspect.getmodule(frame.f_code)
                if module is None:
                    source = inspect.getsource(frame.f_code)
                else:
                    source = inspect.getsource(module)
                current_line = source.split("\n")[frame.f_lineno - 1]

            except OSError as err:
                self.log(f"{err.__class__.__name__}: {err}")
                current_line = ""

            self.log(repr(frame.f_lineno) + " " + current_line)

        if event == "return":
            self.log(frame.f_code.co_name + "()" + " returns " + repr(arg))
            self.last_vars = {}  # Delete 'last' variables


class Debugger(Tracer):
    """Interactive Debugger"""

    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Create a new interactive debugger."""
        self.stepping: bool = True
        self.breakpoints: set[int] = set()
        self.interact: bool = True

        self.frame: FrameType
        self.event: str = ""
        self.arg: Any = None

        self.local_vars: dict[str, Any] = {}

        super().__init__(file=file)

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.frame = frame
        self.local_vars = frame.f_locals  # Dereference exactly once
        self.event = event
        self.arg = arg

        if self.stop_here():
            self.interaction_loop()

    def stop_here(self) -> bool:
        """Return True if we should stop"""
        return self.stepping or self.frame.f_lineno in self.breakpoints

    def interaction_loop(self) -> None:
        """Interact with the user"""
        self.print_debugger_status(self.frame, self.event, self.arg)

        self.interact = True
        while self.interact:
            command = input("(debugger) ")
            self.execute(command)

    def execute(self, command: str) -> None:
        """Execute `command`"""

        sep = command.find(" ")
        if sep > 0:
            cmd = command[:sep].strip()
            arg = command[sep + 1 :].strip()
        else:
            cmd = command.strip()
            arg = ""

        method = self.command_method(cmd)
        if method:
            method(arg)

    def commands(self) -> list[str]:
        """Return a list of commands"""

        cmds = [
            method.replace("_command", "")
            for method in dir(self.__class__)
            if method.endswith("_command")
        ]
        cmds.sort()
        return cmds

    def help_command(self, command: str = "") -> None:
        """Give help on given `command`. If no command is given, give help on all"""

        if command:
            possible_cmds = [
                possible_cmd
                for possible_cmd in self.commands()
                if possible_cmd.startswith(command)
            ]

            if len(possible_cmds) == 0:
                self.log(f"Unknown command {repr(command)}. Possible commands are:")
                possible_cmds = self.commands()
            elif len(possible_cmds) > 1:
                self.log(f"Ambiguous command {repr(command)}. Possible expansions are:")
        else:
            possible_cmds = self.commands()

        for cmd in possible_cmds:
            method = self.command_method(cmd)
            self.log(f"{cmd:10} -- {method.__doc__}")

    def command_method(self, command: str) -> Optional[Callable[[str], None]]:
        """Convert `command` into the method to be called.
        If the method is not found, return `None` instead."""

        if command.startswith("#"):
            return None  # Comment

        possible_cmds = [
            possible_cmd for possible_cmd in self.commands() if possible_cmd.startswith(command)
        ]
        if len(possible_cmds) != 1:
            self.help_command(command)
            return None

        cmd = possible_cmds[0]
        return getattr(self, cmd + "_command")

    def step_command(self, arg: str = "") -> None:
        """Execute up to the next line"""

        self.stepping = True
        self.interact = False

    def continue_command(self, arg: str = "") -> None:
        """Resume execution"""

        self.stepping = False
        self.interact = False

    def print_command(self, arg: str = "") -> None:
        """Print an expression. If no expression is given, print all variables"""

        vars = self.local_vars

        if not arg:
            self.log("\n".join([f"{var} = {repr(value)}" for var, value in vars.items()]))
        else:
            try:
                self.log(f"{arg} = {repr(eval(arg, globals(), vars))}")
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")

    def list_command(self, arg: str = "") -> None:
        """Show current function. If `arg` is given, show its source code."""

        try:
            if arg:
                obj = eval(arg)
                source_lines, line_number = inspect.getsourcelines(obj)
                current_line = -1
            else:
                source_lines, line_number = inspect.getsourcelines(self.frame.f_code)
                current_line = self.frame.f_lineno
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")
            source_lines = []
            line_number = 0
            current_line = -1

        for line in source_lines:
            spacer = " "
            if line_number == current_line:
                spacer = ">"
            elif line_number in self.breakpoints:
                spacer = "#"
            self.log(f"{line_number:4}{spacer} {line}", end="")
            line_number += 1

    def break_command(self, arg: str = "") -> None:
        """Set a breakpoint in given line. If no line is given, list all breakpoints"""

        if arg:
            self.breakpoints.add(int(arg))
        self.log("Breakpoints:", self.breakpoints)

    def delete_command(self, arg: str = "") -> None:
        """Delete breakpoint in line given by `arg`.
        Without given line, clear all breakpoints"""

        if arg:
            try:
                self.breakpoints.remove(int(arg))
            except KeyError:
                self.log(f"No such breakpoint: {arg}")
        else:
            self.breakpoints = set()
        self.log("Breakpoints:", self.breakpoints)

    def quit_command(self, arg: str = "") -> None:
        """Finish execution"""

        self.breakpoints = set()
        self.stepping = False
        self.interact = False

    def assign_command(self, arg: str) -> None:
        """Use as 'assign VAR=VALUE'. Assign VALUE to local variable VAR."""

        sep = arg.find("=")
        if sep > 0:
            var = arg[:sep].strip()
            expr = arg[sep + 1 :].strip()
        else:
            self.help_command("assign")
            return

        vars = self.local_vars
        try:
            vars[var] = eval(expr, self.frame.f_globals, vars)
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")
