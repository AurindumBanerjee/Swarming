#!/usr/bin/env python
"""
new_run.py -- create a self-contained, documented ScratchBench run folder.

Every run gets ITS OWN copies of the code (never an edit of an older file), its own output tree, a launcher,
a manifest and a README; INDEX.md lists all runs. Old runs are never touched.

  Runs/<run_id>/
      code/<part>__<run_id>.py     the script as run (source copy + the explicit settings below, nothing else)
      output/<part>/               ROOT_OUT of that part (written by the run)
      logs/<part>_console.log      console output of that part
      run.sh                       launcher (BLAS pinned to 1 thread, parts run strictly one after the other)
      manifest.json  README.txt    what the run is, md5s of source and as-run code, diff, environment, timeline

Usage (run with the conda "work" python on the server, cwd anywhere):
  python new_run.py create --id run03_original_code --desc "text" \
      --part library=/path/ScratchBench.py --part pure_python=/path/ScratchBench2.py \
      --set 'TARGETS=[0.05, 0.045, 0.04]' \
      --set-part library 'METHODS=["numpy", "solve", "sm", "iterative"]' \
      --set-part pure_python 'METHODS=["pure_python"]'
Only top-level `KEY = value` lines that exist exactly once are replaced; ROOT_OUT is always set per part.
"""
import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

BASE = "/DATA/Aurindum/Swarming"
RUNS = os.path.join(BASE, "MinTime", "Runs")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def md5_text(s):
    return hashlib.md5(s.encode()).hexdigest()


def set_line(text, key, value, where):
    pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    hits = pat.findall(text)
    if len(hits) != 1:
        sys.exit(f"{where}: expected exactly one top-level '{key} = ...' line, found {len(hits)}")
    return pat.sub(lambda m: f"{key} = {value}", text, count=1)


def env_block():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
        except Exception as e:                               # noqa: BLE001
            return f"n/a ({e})"
    import numpy
    import scipy
    return "\n".join([
        f"Created:   {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Machine:   {socket.gethostname()}  ({platform.platform()})",
        "CPU:       " + sh("lscpu | grep 'Model name' | sed 's/.*: *//'"),
        "CPUs:      " + sh("nproc") + " logical; " + sh(
            "lscpu | grep -E '^(Socket|Core|Thread)' | sed 's/ *: */=/' | tr '\\n' ' '"),
        f"Python:    {platform.python_version()}  NumPy {numpy.__version__}  SciPy {scipy.__version__}",
        "BLAS:      " + sh("python -c 'import numpy;numpy.show_config()' 2>&1 | grep -A8 -i 'blas:' "
                           "| grep -E 'name|version' | tr -s ' ' | tr '\\n' ';'"),
        "Threads:   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 (exported by run.sh); particle loop is serial.",
        "Timer:     time.perf_counter() (monotonic wall clock), per FIX 10 in the script",
    ])


def create(a):
    run = os.path.join(RUNS, a.id)
    if os.path.exists(run):
        sys.exit(f"{run} already exists -- runs are never overwritten; choose a new --id")
    glob_set = dict(kv.split("=", 1) for kv in a.set or [])
    part_set = {}
    for name, kv in (a.set_part or []):
        k, v = kv.split("=", 1)
        part_set.setdefault(name, {})[k] = v
    parts = []
    for spec in a.part:
        name, src = spec.split("=", 1)
        parts.append((name, os.path.abspath(src)))
    for sub in ("code", "output", "logs"):
        os.makedirs(os.path.join(run, sub))
    manifest = {"id": a.id, "desc": a.desc, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parts": []}
    diffs = []
    for name, src in parts:
        text = open(src, newline="").read().replace("\r\n", "\n")
        new = text
        root_out = f"MinTime/Runs/{a.id}/output/{name}"
        settings = {"ROOT_OUT": json.dumps(root_out)}
        settings.update(glob_set)
        settings.update(part_set.get(name, {}))
        for k, v in settings.items():
            new = set_line(new, k, v, f"{name} ({src})")
        dst = os.path.join(run, "code", f"{name}__{a.id}.py")
        open(dst, "w", newline="").write(new)
        d = "".join(difflib.unified_diff(text.splitlines(1), new.splitlines(1),
                                         os.path.basename(src), os.path.basename(dst), n=0))
        diffs.append(f"--- {name}\n{d or '(identical to source)'}")
        manifest["parts"].append({"name": name, "source": src, "source_md5": md5(src),
                                  "code": os.path.relpath(dst, run), "code_md5": md5(dst),
                                  "settings_applied": settings})
    # launcher
    sh = ["#!/bin/bash",
          f"# Run {a.id}: parts execute strictly one after the other. cwd must be {BASE} (ROOT_OUT is relative).",
          f"RUN={run}", f"cd {BASE} || exit 1",
          "source /DATA/Aurindum/conda/etc/profile.d/conda.sh", "conda activate work",
          "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
          "stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }", ""]
    for p in manifest["parts"]:
        n = p["name"]
        sh += [f'echo "Part {n} START: $(stamp)  load: $(cut -d" " -f1-3 /proc/loadavg)" >> "$RUN/README.txt"',
               f'python -u "$RUN/{p["code"]}" > "$RUN/logs/{n}_console.log" 2>&1',
               f'echo "Part {n} END:   $(stamp)  exit code $?" >> "$RUN/README.txt"', ""]
    open(os.path.join(run, "run.sh"), "w", newline="").write("\n".join(sh))
    os.chmod(os.path.join(run, "run.sh"), 0o755)
    json.dump(manifest, open(os.path.join(run, "manifest.json"), "w"), indent=2)
    with open(os.path.join(run, "README.txt"), "w") as fh:
        fh.write(f"RUN {a.id}\n{a.desc}\n\n{env_block()}\n\nParts / code actually run:\n")
        for p in manifest["parts"]:
            fh.write(f"  {p['name']:<12} {p['code']}  md5 {p['code_md5']}  (source {p['source']} md5 {p['source_md5']})\n")
        fh.write("\nChanges vs source (only these lines differ):\n" + "\n".join(diffs) + "\n")
        fh.write("\nLaunch: cd " + BASE + " && nohup setsid bash " + run + "/run.sh > " + run
                 + "/logs/driver.log 2>&1 < /dev/null &\n\n---- timeline ----\n")
    idx = os.path.join(RUNS, "INDEX.md")
    if not os.path.exists(idx):
        open(idx, "w").write("# ScratchBench runs\n\n| run | what | code md5 (as run) | location |\n|---|---|---|---|\n")
    with open(idx, "a") as fh:
        fh.write(f"| {a.id} | {a.desc} | " + ", ".join(f"{p['name']} {p['code_md5'][:8]}" for p in manifest["parts"])
                 + f" | Runs/{a.id}/ |\n")
    print(f"created {run}")
    for p in manifest["parts"]:
        print(f"  {p['code']}  md5 {p['code_md5']}")
    print("\n".join(diffs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("create")
    c.add_argument("--id", required=True)
    c.add_argument("--desc", required=True)
    c.add_argument("--part", action="append", required=True, help="NAME=source_script.py (repeatable)")
    c.add_argument("--set", action="append", help="KEY=VALUE applied to every part")
    c.add_argument("--set-part", action="append", nargs=2, metavar=("NAME", "KEY=VALUE"))
    c.set_defaults(fn=create)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
