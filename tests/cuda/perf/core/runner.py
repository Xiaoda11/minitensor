"""CPU timing utility — useful when GPU is unavailable."""

import subprocess
import time


def run_cpu(cmd: list[str]) -> float:
    """Run a command and return wall-clock time in seconds.

    Args:
        cmd: Command as a list (e.g. ['./my_binary', '--flag']).

    Returns:
        Elapsed wall-clock time in seconds.
    """
    t0 = time.time()
    subprocess.check_call(cmd)
    return time.time() - t0
