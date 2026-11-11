"""Process orchestration shared by the main benchmark runner."""

import subprocess
from time import sleep

from replay import ROOT
from run_qcad_baselines import METHODS as POINT_AND_CONTEXT_METHODS


CLASSICAL_METHODS = (*POINT_AND_CONTEXT_METHODS, "CAD")


def run_groups(commands, work, env):
    active = {}
    failed = []
    try:
        for group, command in commands.items():
            logfile = work / f"{group}.log"
            with logfile.open("a") as log:
                active[group] = subprocess.Popen(
                    command, cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            print(f"Started {group} (PID {active[group].pid}); {logfile}", flush=True)
        while active:
            for group, process in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[group]
                print(f"{group} exited with status {status}", flush=True)
                if status:
                    failed.append(group)
            if active:
                sleep(1)
    finally:
        for process in active.values():
            process.terminate()
        for process in active.values():
            process.wait()
    if failed:
        raise SystemExit(f"See benchmark logs for failed groups: {', '.join(failed)}")
