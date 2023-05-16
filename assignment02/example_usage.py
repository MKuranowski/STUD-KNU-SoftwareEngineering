from utils import clear_next_inputs, next_inputs
from debugger import Debugger
from sha256 import generate_hash

clear_next_inputs()
next_inputs(["step", "step", "break 101", "continue", "step", "continue"])
with Debugger():
    encoded = "debuggggggggggggggg".encode()
    hash = generate_hash(encoded).hex()
