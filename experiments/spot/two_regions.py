"""Run a spot.py command, or a script that imports spot (peek.py), across two Nebius regions. Phase d5 used it:
bpb-d5-0 (extended arm) lived in eu-west1 and bpb-d5-1 (stock arm) in us-central1, so each waited on its own
region's price. New machines are created in us-central1; listing, watch, push and delete see both projects
(instance ids are global). The project and subnet ids come from .env (NEBIUS_PROJECT_EU_WEST1,
NEBIUS_PROJECT_US_CENTRAL1, NEBIUS_SUBNET_US_CENTRAL1), kept out of this public repo.
The cap is d5's, raised step by step from $1.50 to $2.50 with the user's OK for that run only; spot.py's own
default stays at the standing $0.99.
Run from uraikal/: python3 experiments/spot/two_regions.py up d5 2 --gpu h200 | watch d5 | experiments/spot/peek.py d5"""
import runpy
import sys
import time
from decimal import Decimal

sys.path.insert(0, "experiments/spot")
import spot

base = spot.env
WEST, CENTRAL = base()["NEBIUS_PROJECT_EU_WEST1"], base()["NEBIUS_PROJECT_US_CENTRAL1"]
spot.env = lambda: {**base(), "NEBIUS_PROJECT": CENTRAL, "NEBIUS_SUBNET": base()["NEBIUS_SUBNET_US_CENTRAL1"]}
spot.MAX_PRICE = spot.CEILING = Decimal("2.500")  # d5 only
spot.machines = lambda phase: [m for p in (WEST, CENTRAL)
                               for m in spot.nebius("compute", "instance", "list", "--parent-id", p).get("items", [])
                               if m["metadata"]["name"].startswith(f"bpb-{phase}-")]
_nebius = spot.nebius


def start_or_skip(*args):
    """watch: one start refused on price ends its whole pass, so the other region's machine never got its turn."""
    try:
        return _nebius(*args)
    except SystemExit as err:
        if args[:3] != ("compute", "instance", "start"):
            raise
        print(time.strftime("%H:%M"), "start refused:", args[-1], str(err)[-120:].replace("\n", " "), flush=True)
        return {}


spot.nebius = start_or_skip
if sys.argv[1].endswith(".py"):
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")  # its `import spot` gets this patched module
else:
    spot.main()
