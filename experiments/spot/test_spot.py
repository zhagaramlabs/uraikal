"""Self-checks for spot.py's machine lifecycle (no cloud calls). Run: python3 experiments/spot/test_spot.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spot import action, ls_names

# Every row of action()'s table, as watch behaved before the table existed.
for state in ("RUNNING", "STOPPED", "STARTING", "STOPPING", "?"):
    assert action(state, finished=True, failed=False) == ("stop" if state == "RUNNING" else None), state
    assert action(state, finished=True, failed=True) == ("stop" if state == "RUNNING" else None), state
    assert action(state, finished=False, failed=True) is None, state
    assert action(state, finished=False, failed=False) == ("start" if state == "STOPPED" else None), state

# Markers match exactly: worker-1 isn't done because worker-10 is (watch used a substring test).
listing = """2026-11-07 10:00:00          5 worker-10
2026-11-07 10:00:00        812 failed-11
                           PRE a-truncated/
"""
got = ls_names(listing)
assert got == {"worker-10", "failed-11", "a-truncated/"}, got
assert "worker-1" not in got and "failed-1" not in got and ls_names("") == set()
print("all checks passed")
