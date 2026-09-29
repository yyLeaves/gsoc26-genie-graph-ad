"""Queue regression tests use dummy children, never training jobs or GPUs."""

from pathlib import Path
import re
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SHELL_QUEUES = ("run_gladc_formal.sh", "run_gladc_eb2.sh")


def wait_blocks():
    for name in SHELL_QUEUES:
        source = (SCRIPTS / name).read_text()
        blocks = re.findall(r"^wait [^\n]+$|^failed=0\n.*?^fi$",
                            source, flags=re.MULTILINE | re.DOTALL)
        assert blocks, f"No wait block tested in {name}"
        for index, block in enumerate(blocks):
            yield pytest.param(block, id=f"{name}-{index}")


@pytest.mark.parametrize("block", list(wait_blocks()))
@pytest.mark.parametrize("failure", [None, "first", "middle", "last"])
def test_shell_queue_checks_every_child_and_waits_for_all(tmp_path, block, failure):
    variables = list(dict.fromkeys(re.findall(r"\$\{([pe]\d+)\}", block)))
    assert variables
    failed_index = {None: -1, "first": 0, "middle": len(variables) // 2,
                    "last": len(variables) - 1}[failure]
    launches = []
    for index, variable in enumerate(variables):
        code = 7 if index == failed_index else 0
        launches.append(
            f"(sleep 0.05; touch done_{index}; exit {code}) & {variable}=$!")
    result = subprocess.run(
        ["bash", "-c", "\n".join(["set -euo pipefail", *launches, block,
                                    "echo completed"])],
        cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) == (failure is None), result.stdout
    assert ("completed" in result.stdout) == (failure is None)
    assert len(list(tmp_path.glob("done_*"))) == len(variables)
