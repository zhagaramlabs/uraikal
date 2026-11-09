"""Run a phase of tokenizer_bpb.py jobs on Nebius preemptible GPU machines. Runs on the Mac.

    python experiments/spot/spot.py pack            # code + data to R2 (rerun after changing code or jobs)
    python experiments/spot/spot.py up a 2          # 2 preemptible H100 machines for jobs_a.txt
    python experiments/spot/spot.py watch a         # every 5 min: restart preempted machines, stop finished ones; ends when all are
    python experiments/spot/spot.py watch a --once  # one status pass
    python experiments/spot/spot.py stop a          # stop every machine of the phase
    python experiments/spot/spot.py delete a        # delete them and their disks; refuses unless all reported done

Each machine runs worker.sh at every boot and takes every WORKERS-th line of jobs_<phase>.txt. A preempted
machine is started again, not deleted: its disk keeps the data and checkpoints. Results land in
r2:zhagaram-data/tokenizer/results/<phase>/. Machines get only the bucket-scoped key (R2_ZD_* in .env); the
Mac uses the same key, for pack and status.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

MAX_PRICE = Decimal("0.990")  # USD per GPU-hour, the user's hard ceiling for every new machine (2026-11-06). Never raise it without asking.
CEILING = Decimal("0.99")
assert MAX_PRICE <= CEILING

HERE = Path(__file__).resolve().parent
URAIKAL = HERE.parents[1]
ENV = URAIKAL.parent / ".env"
S3 = "s3://zhagaram-data/tokenizer"
SSH = ["-i", str(Path.home() / ".ssh/zhagaram_nebius"), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
       "-o", "ConnectTimeout=10"]


def env():  # also holds the Nebius project and subnet ids (NEBIUS_PROJECT, NEBIUS_SUBNET), kept out of this public repo
    pairs = (line.split("=", 1) for line in ENV.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip().strip("'\"") for k, v in pairs}


def nebius(*args):
    out = subprocess.run([str(Path.home() / ".nebius/bin/nebius"), "--profile", "zhagaram-sa", *args, "--format", "json"],
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"nebius {args[:3]} failed: {out.stderr.strip()[:500]}")
    return json.loads(out.stdout or "{}")


def r2(*args, **kw):
    e = env()
    keys = {"AWS_ACCESS_KEY_ID": e["R2_ZD_ACCESS_KEY"], "AWS_SECRET_ACCESS_KEY": e["R2_ZD_SECRET_ACCESS_KEY"], "AWS_DEFAULT_REGION": "auto"}
    return subprocess.run(["aws", "--endpoint-url", e["R2_ZD_S3_ENDPOINT"], "s3", *args], env={**os.environ, **keys}, **kw)


def checked_price(policy):
    """The policy's max price, if it is within the ceiling; exits otherwise."""
    price = Decimal(policy["spec"]["pricing"]["max_price_v1"]["max_price"])
    if price > CEILING:
        sys.exit(f"pricing policy {policy['metadata']['name']} allows ${price}/GPU-h, above the ${CEILING} ceiling")
    return price


def pricing_policy(gpu):
    """Id of the capped policy for this GPU, created on first use. Checks the cap as Nebius stores it, not as we sent it."""
    name = f"zhagaram-{gpu}-max-{MAX_PRICE}"
    found = [p for p in nebius("billing", "pricing-policy", "list", "--parent-id", env()["NEBIUS_PROJECT"]).get("items", [])
             if p["metadata"]["name"] == name]
    policy = found[0] if found else nebius(
        "billing", "pricing-policy", "create", "--parent-id", env()["NEBIUS_PROJECT"], "--name", name,
        "--compute-instance-spec-v1-platform", f"gpu-{gpu}-sxm", "--pricing-max-price-v1-max-price", str(MAX_PRICE))
    print(f"pricing policy {name}: max ${checked_price(policy)}/GPU-h,",
          policy.get("status", {}).get("scheduling_state", "?"), flush=True)  # BLOCKED = the market is above the cap now
    return policy["metadata"]["id"]


def machines(phase):
    return [m for m in nebius("compute", "instance", "list", "--parent-id", env()["NEBIUS_PROJECT"]).get("items", [])
            if m["metadata"]["name"].startswith(f"bpb-{phase}-")]


def ip(m):
    nics = m.get("status", {}).get("network_interfaces", [])
    return nics[0].get("public_ip_address", {}).get("address", "").split("/")[0] if nics else ""


def pack(_args):
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=URAIKAL,
                           capture_output=True, text=True, check=True).stdout.split()
    subprocess.run(["tar", "czf", "/tmp/zl-code.tgz", *files], cwd=URAIKAL, check=True)
    sums = subprocess.run("shasum -a 256 bpb*/train.jsonl.gz bpb*/eval.json $(ls bpb*/tasks.json 2>/dev/null) "
                          "tamil_tokenizer/*.json hf_tokenizers/*/*",
                          shell=True, cwd=URAIKAL / "data", capture_output=True, text=True, check=True).stdout
    (Path("/tmp/zl-SHA256SUMS")).write_text(sums)
    r2("cp", "/tmp/zl-code.tgz", f"{S3}/code.tgz", "--only-show-errors", check=True)
    r2("cp", "/tmp/zl-SHA256SUMS", f"{S3}/SHA256SUMS", "--only-show-errors", check=True)
    for d in ("hf_tokenizers", "tamil_tokenizer"):  # SHA256SUMS lists every tokenizer, so each must be in R2
        r2("sync", str(URAIKAL / "data" / d), f"{S3}/{d}", "--only-show-errors", check=True)
    for d in sorted((URAIKAL / "data").glob("bpb*")):  # text and test sets; each machine builds its own token caches
        r2("sync", str(d), f"{S3}/{d.name}", "--exclude", "tokens/*", "--exclude", "models/*", "--only-show-errors", check=True)
    print(f"packed {len(files)} code files and {len(sums.splitlines())} checksums")


def up(args):
    e = env()
    if not e.get("R2_ZD_ACCESS_KEY") or not e.get("R2_ZD_SECRET_ACCESS_KEY"):
        sys.exit("no R2_ZD_ACCESS_KEY / R2_ZD_SECRET_ACCESS_KEY in .env: machines only get a key limited to zhagaram-data")
    policy_id = pricing_policy(args.gpu)  # new machines never follow the open spot price: Nebius stops them above the cap
    key = (Path.home() / ".ssh/zhagaram_nebius.pub").read_text().strip()
    worker = base64.b64encode((HERE / "worker.sh").read_bytes()).decode()
    cloud = f"""#cloud-config
users:
  - name: zl
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys: ["{key}"]
write_files:
  - path: /root/worker.sh
    permissions: "0755"
    encoding: b64
    content: {worker}
  - path: /etc/systemd/system/spot-worker.service
    content: |
      [Unit]
      Wants=network-online.target
      After=network-online.target
      [Service]
      ExecStart=/bin/bash /root/worker.sh
      Restart=on-failure
      RestartSec=60
      [Install]
      WantedBy=multi-user.target
runcmd:
  - [systemctl, daemon-reload]
  - [systemctl, enable, --now, spot-worker]
"""
    names = [f"bpb-{args.phase}-{k}" for k in range(args.n)]
    existing = {m["metadata"]["name"] for m in machines(args.phase)}
    left = markers("done", args.phase) | markers("logs", args.phase)
    stale = [n for n in names if n not in existing and {f"worker-{worker(n)}", f"failed-{worker(n)}"} & left]
    if stale:  # watch would stop a new machine at once, or never restart it, on an old run's marker
        sys.exit(f"{', '.join(stale)}: an earlier run of phase {args.phase} left done/ or failed- markers in R2. "
                 f"Use a new phase name, or move {S3}/done/{args.phase}/ and its failed-* logs aside first")
    for name in (n for n in names if n not in existing):  # a rerun after a partial failure skips what exists
        nebius("compute", "instance", "create", "--parent-id", env()["NEBIUS_PROJECT"], "--name", name,
               "--resources-platform", f"gpu-{args.gpu}-sxm", "--resources-preset", "1gpu-16vcpu-200gb",
               "--preemptible-on-preemption", "STOP", "--spot-pricing-policy-id", policy_id, "--recovery-policy", "fail",
               "--boot-disk-attach-mode", "READ_WRITE", "--boot-disk-managed-disk-name", f"{name}-boot",
               "--boot-disk-managed-disk-type", "network_ssd",
               "--boot-disk-managed-disk-size-gibibytes", str(args.disk),
               "--boot-disk-managed-disk-source-image-family-image-family", "ubuntu24.04-cuda13.0",
               "--network-interfaces", json.dumps([{"name": "eth0", "subnet_id": env()["NEBIUS_SUBNET"], "ip_address": {}, "public_ip_address": {}}]),
               "--cloud-init-user-data", cloud)
        print("created", name, flush=True)
    spot_env = (f"R2_S3_ENDPOINT={e['R2_ZD_S3_ENDPOINT']}\nAWS_ACCESS_KEY_ID={e['R2_ZD_ACCESS_KEY']}\n"
                f"AWS_SECRET_ACCESS_KEY={e['R2_ZD_SECRET_ACCESS_KEY']}\nPHASE={args.phase}\nWORKERS={args.n}\n")
    for k, name in enumerate(names):
        for _ in range(60):  # up to 30 min for the machine and its ssh to come up
            m = next((m for m in machines(args.phase) if m["metadata"]["name"] == name), None)
            if m and ip(m) and subprocess.run(["ssh", *SSH, f"zl@{ip(m)}", "umask 077 && cat > spot.env"],
                                              input=spot_env + f"WORKER={k}\n", text=True, capture_output=True).returncode == 0:
                print(name, "running at", ip(m), flush=True)
                break
            time.sleep(30)
        else:
            sys.exit(f"{name} never took its spot.env: check it with `watch {args.phase} --once`")


def ls_names(listing):
    """Object names in `aws s3 ls` output (date, time, size, name per line), as a set for exact matching:
    a substring test would count worker-1 as done once worker-10 is."""
    return {line.split()[-1] for line in listing.splitlines() if line.strip()}


def markers(kind, phase):
    """The phase's done/ or logs/ objects in R2. An R2 error raises: an empty answer would read as "nothing is
    done", and watch would start every finished machine again."""
    out = r2("ls", f"{S3}/{kind}/{phase}/", capture_output=True, text=True)
    if out.returncode not in (0, 1):  # 1: nothing there yet
        raise RuntimeError(f"R2 ls {kind}/{phase}/ failed: {out.stderr.strip()[:300]}")
    return ls_names(out.stdout)


def worker(name):
    return name.rsplit("-", 1)[1]


def action(state, finished, failed):
    """What watch does with one machine. Its state is Nebius's; its worker's lifecycle lives in R2 markers.

    | Worker (R2 marker)            | Nebius state      | watch does                                  |
    |-------------------------------|-------------------|---------------------------------------------|
    | finished (done/<p>/worker-k)  | RUNNING           | stop: it powers itself off; this makes sure |
    | finished                      | anything else     | nothing                                     |
    | failed (logs/<p>/failed-k)    | anything          | nothing: left stopped for a look            |
    | neither                       | STOPPED           | start: it was preempted                     |
    | neither                       | anything else     | nothing: booting, working or stopping       |
    """
    if finished:
        return "stop" if state == "RUNNING" else None
    return "start" if state == "STOPPED" and not failed else None


def has_settings(m):
    """False only if ssh works and spot.env is missing: the worker then waits forever on a billing GPU."""
    run = subprocess.run(["ssh", *SSH, f"zl@{ip(m)}", "test -f spot.env"], capture_output=True)
    return run.returncode != 1


def watch(args):
    while True:
        try:
            done, failed = markers("done", args.phase), markers("logs", args.phase)
            ms, ended = machines(args.phase), 0
            for m in ms:
                name, state = m["metadata"]["name"], m.get("status", {}).get("state", "?")
                finished, broke = f"worker-{worker(name)}" in done, f"failed-{worker(name)}" in failed
                todo = None if args.once else action(state, finished, broke)
                if todo:
                    nebius("compute", "instance", todo, "--id", m["metadata"]["id"])
                    state = {"stop": "stopping (finished)", "start": "starting again (was preempted)"}[todo]
                ended += finished and state == "STOPPED"
                stuck = state == "RUNNING" and not finished and not broke and ip(m) and not has_settings(m)
                print(time.strftime("%H:%M"), name, state, ip(m), "done" if finished else "", "FAILED" if broke else "",
                      f"NO SETTINGS: rerun `up {args.phase} <n>`" if stuck else "", flush=True)
            if ms and ended == len(ms):
                print(time.strftime("%H:%M"), f"all {len(ms)} machines finished and stopped: watch ends", flush=True)
                return
        except (SystemExit, Exception) as err:  # one failed API call must not end a 12-hour watch
            print(time.strftime("%H:%M"), "watch error, retrying in 5 min:", err, flush=True)
        if args.once:
            return
        time.sleep(300)


def stop(args):
    for m in machines(args.phase):
        nebius("compute", "instance", "stop", "--id", m["metadata"]["id"])
        print("stopping", m["metadata"]["name"])


def delete(args):
    """Delete a phase's machines with their disks, only if every worker reported done. A worker reports done
    only after each of its results is in R2, so nothing on a deleted disk is lost."""
    done = markers("done", args.phase)
    ms = machines(args.phase)
    unfinished = [m["metadata"]["name"] for m in ms if f"worker-{worker(m['metadata']['name'])}" not in done]
    if unfinished:
        sys.exit(f"not deleting anything: {', '.join(unfinished)} never reported done, so R2 may lack some results")
    for m in ms:
        nebius("compute", "instance", "delete", "--id", m["metadata"]["id"])
        print("deleted", m["metadata"]["name"], "and its boot disk")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack")
    a = sub.add_parser("up")
    a.add_argument("phase")
    a.add_argument("n", type=int)
    a.add_argument("--gpu", choices=["h100", "h200"], default="h100")
    a.add_argument("--disk", type=int, default=200, help="boot disk, GiB")
    a = sub.add_parser("watch")
    a.add_argument("phase")
    a.add_argument("--once", action="store_true")
    a = sub.add_parser("stop")
    a.add_argument("phase")
    a = sub.add_parser("delete")
    a.add_argument("phase")
    args = p.parse_args()
    {"pack": pack, "up": up, "watch": watch, "stop": stop, "delete": delete}[args.cmd](args)


if __name__ == "__main__":
    main()
