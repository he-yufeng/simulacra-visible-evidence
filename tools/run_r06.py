"""One fixed public-synthetic r05/r06 comparison. Never an official submission."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "_reference"
COMMIT = "0d2332d8ae19a8ce171031142bdc97134910e7ec"
R05_SHA = "86d0eb535267f338e05e31aae8585198f953202941e1a7cc03fddccfcffc1184"
R06_SHA = "79ac3ad116075f069cc2d8ead7724649445600649f212a19dc9af3466c292a01"
FIRST_DATA_SHA = "cc331049335fa996c4cb24424cba71595a61e4690635e9fddc949abea2dc9bb5"
SEEDS = (202610061, 202610062)
INSTRUMENTS = ("unhcr", "unicef", "world_bank")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    files = [ROOT / name for name in ("tools/run_r06.py", "R06_PROTOCOL.md",
             "tests/test_participant_r06.py", "benchmark-requirements.txt")]
    files += [ROOT / variant / name for variant in ("participant_r05", "participant_r06")
              for name in ("main.py", "requirements.txt", "LICENSE")]
    files += [REFERENCE / name for name in ("score.py", "make_sandbox.py", "config.yml",
              "LICENSE", "tools/check_submission_zip.py")]
    files += [REFERENCE / "data" / (name + ".json") for name in INSTRUMENTS]
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("missing or linked input")
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def main():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("protocol requires Python3.12")
    head = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"],
                                   text=True, timeout=15).strip()
    dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"],
                                    text=True, timeout=15).strip()
    if head != COMMIT or dirty:
        raise ValueError("official checkout must be exact and clean")
    if digest(ROOT / "participant_r05/main.py") != R05_SHA or digest(ROOT / "participant_r06/main.py") != R06_SHA:
        raise ValueError("predictor source differs from frozen protocol")
    frozen = inputs()
    out = ROOT / "_bench"
    out.mkdir(mode=0o700)
    started = time.monotonic()
    summary = {"scope": "PUBLIC_SYNTHETIC_METHOD_NOT_OFFICIAL_SCORE",
               "source_commit": COMMIT, "inputs": frozen, "seeds": SEEDS,
               "cells": [], "pairs": [], "dataset_hashes": {},
               "archives": [], "processes": [], "feasibility_gate": False,
               "official_upload_performed": False, "private_microdata_used": False,
               "predict_cap_seconds": 300, "three_instrument_predict_cap_sum": 900,
               "first_failed_run": 37172888118}

    def save():
        (out / "receipt.json").write_text(json.dumps(summary, sort_keys=True, indent=2))

    def run(command, label, cap=420):
        remaining = 1800 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("fixed batch timeout")
        if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 10 * 1024**3:
            raise RuntimeError("remote free disk below10GiB")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 50 * 1024**2:
            raise RuntimeError("remote output exceeds50MiB")
        step_started = time.monotonic()
        with subprocess.Popen(command, cwd=REFERENCE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, start_new_session=True) as p:
            try:
                output, _ = p.communicate(timeout=min(cap, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    output, _ = p.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    output, _ = p.communicate()
                (out / (label + ".txt")).write_text(output)
                raise TimeoutError("owned step timed out")
        log = out / (label + ".txt")
        log.write_text(output)
        summary["processes"].append({"label": label, "exit_code": p.returncode,
            "wall_seconds": time.monotonic()-step_started, "log_sha256": digest(log)})
        save()
        if p.returncode:
            print("STEP_FAILED " + label + " " + output[-1500:], flush=True)
            raise RuntimeError("owned step failed")

    def dataset_hashes(directory):
        return {str(p.relative_to(out)): digest(p) for p in sorted(directory.iterdir()) if p.is_file()}

    exit_code = 1
    try:
        run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
             "-p", "test_participant_r06.py", "-v"], "new_linux_microtests", cap=45)
        for variant in ("r05", "r06"):
            archive = out / (variant + ".zip")
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
                for name in ("main.py", "requirements.txt", "LICENSE"):
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.external_attr = 0o100644 << 16
                    z.writestr(entry, (ROOT / ("participant_" + variant) / name).read_bytes())
            run([sys.executable, str(REFERENCE / "tools/check_submission_zip.py"), str(archive)],
                "archive_" + variant, cap=30)
            summary["archives"].append({"variant": variant, "sha256": digest(archive),
                                        "bytes": archive.stat().st_size, "status": "PASS"})
        print("NEW_LINUX_MICROTESTS_AND_ARCHIVE_GATES_PASS", flush=True)
        stop = False
        for seed in SEEDS:
            for instrument in INSTRUMENTS:
                schema = REFERENCE / "data" / (instrument + ".json")
                dataset = out / f"data_{seed}_{instrument}"
                run([sys.executable, str(REFERENCE / "make_sandbox.py"), "--schema", str(schema),
                     "--config", str(REFERENCE / "config.yml"), "--seed", str(seed), "--out", str(dataset)],
                    f"generate_{seed}_{instrument}")
                original_data = dataset_hashes(dataset)
                if seed == SEEDS[0] and instrument == "unhcr":
                    if digest(dataset / "respondents.parquet") != FIRST_DATA_SHA:
                        raise RuntimeError("first attempted deterministic dataset differs; no scoring")
                summary["dataset_hashes"].update(original_data)
                paired = {}
                for variant in ("r05", "r06"):
                    if dataset_hashes(dataset) != original_data:
                        raise RuntimeError("paired data changed")
                    log = out / f"{seed}_{instrument}_{variant}.score.json"
                    run([sys.executable, str(REFERENCE / "score.py"), "--submission",
                         str(ROOT / ("participant_" + variant)), "--data", str(dataset),
                         "--schema", str(schema), "--config", str(REFERENCE / "config.yml"),
                         "--phase", "1", "--seed", str(seed), "--timeout", "300", "--log", str(log)],
                        f"score_{seed}_{instrument}_{variant}")
                    result = json.loads(log.read_text())
                    if result.get("status") != "PASS" or not math.isfinite(result["skill"]):
                        raise RuntimeError("invalid official local score result")
                    if dataset_hashes(dataset) != original_data:
                        raise RuntimeError("data changed during score")
                    cell = {key: result[key] for key in ("skill", "reported_skill", "n_cells",
                            "n_held_out", "n_respondents", "log_score")}
                    cell.update(seed=seed, instrument=instrument, variant=variant,
                                score_log_sha256=digest(log), status="PASS")
                    paired[variant] = cell
                    summary["cells"].append(cell)
                    print("CELL_JSON " + json.dumps(cell, sort_keys=True), flush=True)
                for key in ("n_cells", "n_held_out", "n_respondents"):
                    if paired["r05"][key] != paired["r06"][key]:
                        raise RuntimeError("paired denominators differ")
                pair = {"seed": seed, "instrument": instrument,
                        "delta_skill": paired["r06"]["skill"]-paired["r05"]["skill"]}
                summary["pairs"].append(pair)
                print("PAIR_JSON " + json.dumps(pair, sort_keys=True), flush=True)
                wins = sum(p["delta_skill"] >= .005 for p in summary["pairs"])
                regressions = sum(p["delta_skill"] < -.015 for p in summary["pairs"])
                if wins + (6-len(summary["pairs"])) < 4 or regressions:
                    summary.update(status="STOP_FUTILITY", useful_pairs=wins,
                                   material_regression_pairs=regressions,
                                   unstarted_pairs=6-len(summary["pairs"]))
                    stop = True
                    break
            if stop:
                break
        if not stop:
            wins = sum(p["delta_skill"] >= .005 for p in summary["pairs"])
            regressions = sum(p["delta_skill"] < -.015 for p in summary["pairs"])
            summary.update(status="PASS", useful_pairs=wins, material_regression_pairs=regressions,
                           feasibility_gate=len(summary["cells"])==12 and wins>=4 and regressions==0)
        exit_code = 0
    except Exception as exc:
        summary.update(status="FAIL", error_class=type(exc).__name__)
    summary["frozen_inputs_unchanged"] = inputs() == frozen
    summary["wall_seconds"] = time.monotonic()-started
    if not summary["frozen_inputs_unchanged"]:
        summary.update(status="FAIL", feasibility_gate=False)
        exit_code = 1
    save()
    print("FINAL_RECEIPT_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
