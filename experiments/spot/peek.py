"""One status line per machine of a phase: state, setup, the newest log line of each job, failures.
Never prints a machine's address. Run from uraikal/: python3 experiments/spot/peek.py <phase>"""
import subprocess
import sys

sys.path.insert(0, "experiments/spot")
import spot

phase = sys.argv[1]
ms = sorted(spot.machines(phase), key=lambda m: m["metadata"]["name"])
if not ms:
    print(f"no {phase} machines")
remote = ("sudo bash -c 'cd /root; [ -f ready ] && echo setup:ready || echo setup:fetching; "
          "for f in logs/qd-*.log; do [ -f \"$f\" ] && echo \"$f: $(grep -E \"step|Error|error|Traceback|Killed|vocab\" \"$f\" | grep -v _POSIX | tail -n 1 | cut -c1-200)\"; done; "
          "[ -f logs/failed ] && echo FAILED: $(cat logs/failed); true'")
for m in ms:
    state, addr = m.get("status", {}).get("state", "?"), spot.ip(m)
    detail = ""
    if addr and state == "RUNNING":
        try:
            out = subprocess.run(["ssh", *spot.SSH, f"zl@{addr}", remote], capture_output=True, text=True, timeout=60)
            detail = " | ".join(out.stdout.split("\n")).strip(" |") if out.returncode == 0 else "ssh failed"
        except subprocess.TimeoutExpired:
            detail = "ssh timed out"
    line = f"{m['metadata']['name']} {state} {detail}"
    print(line.replace(addr, "<ip>") if addr else line)
