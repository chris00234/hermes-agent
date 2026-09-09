"""Standalone Lane A verification without repository conftest side effects."""

import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "agent/memory_manager.py", root / "agent/memory_lifecycle.py"]
    paths.extend(sorted((root / "tests/agent").glob("test_lane_a_memory*.py")))
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        print(f"In-memory compile OK: {path.relative_to(root)}", flush=True)
    from agent.memory_manager import MemoryManager

    print(f"Real import OK: {MemoryManager.__module__}", flush=True)
    if sys.argv[1:] == ["compile"]:
        return 0
    import pytest

    return pytest.main([
        "--noconftest", "-c", "/dev/null", "-p", "no:cacheprovider", *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
