"""One frozen Linux exact-ZIP public-synthetic method comparison."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "_reference"
OUT = ROOT / "_method_r11"
COMMIT = "0d2332d8ae19a8ce171031142bdc97134910e7ec"
R05_SHA = "86d0eb535267f338e05e31aae8585198f953202941e1a7cc03fddccfcffc1184"
R11_SHA = "32f16501feeb7b6890262165d3f510b14b818208e0ee73aca716d847ead6cdf8"
SEEDS = (202610121, 202610122)
INSTRUMENTS = ("unhcr", "unicef", "world_bank")
CAPS = {"unhcr": 480, "unicef": 120, "world_bank": 240}
PAYLOAD = ("main.py", "requirements.txt", "LICENSE")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (sys.version_info[:2] != (3, 12) or platform.system() != "Linux"
            or platform.machine() != "x86_64"):
        raise ValueError("frozen method requires native Linux x86_64 Python3.12")
    manifest = json.loads((ROOT / "R11_SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["inputs"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("frozen participant input mismatch")
    head = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"], text=True).strip()
    if head != COMMIT or dirty:
        raise ValueError("official checkout must be exact and clean")
    if sha(ROOT / "participant_r05/main.py") != R05_SHA or sha(ROOT / "participant_r11/main.py") != R11_SHA:
        raise ValueError("frozen model mismatch")
    OUT.mkdir(mode=0o700)
    started = time.monotonic()
    report = {"scope": "INDEPENDENT_LINUX_EXACT_ZIP_PUBLIC_SYNTHETIC_METHOD_NOT_OFFICIAL_SCORE",
        "status": "RUNNING", "source_commit": os.environ.get("GITHUB_SHA"),
        "repository": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "official_commit": COMMIT, "inputs": manifest["inputs"],
        "official_files": {name: sha(REFERENCE / name) for name in
                           ("score.py", "make_sandbox.py", "config.yml", "LICENSE")},
        "runtime": {"python": platform.python_version(), "machine": platform.machine(),
                    "system": platform.system(), "versions": {name: metadata.version(name)
                    for name in ("numpy", "pandas", "catboost", "scipy", "pyarrow", "PyYAML")}},
        "seeds": SEEDS, "cells": [], "pairs": [], "processes": [], "archives": [],
        "dataset_hashes": {}, "candidate_parent_seconds_by_seed": {}, "feasibility_gate": False,
        "official_score": None, "official_upload_performed": False, "private_microdata_used": False,
        "gpu_h100_equivalence": False, "new_service_spend_cny": 0}

    def save():
        (OUT / "receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2))

    def run(command, label, cap):
        remaining = 1800 - (time.monotonic() - started)
        disk = os.statvfs(ROOT)
        if remaining <= 0 or disk.f_bavail * disk.f_frsize < 10 * 1024**3:
            raise RuntimeError("study time/disk resource gate")
        if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > 50 * 1024**2:
            raise RuntimeError("study output exceeds50MiB")
        step = time.monotonic()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PIP_NO_CACHE_DIR="1",
                           PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", MPLCONFIGDIR=str(OUT / "mpl_config"))
        with subprocess.Popen(command, cwd=REFERENCE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as child:
            timeout = False
            try:
                output, _ = child.communicate(timeout=min(cap, remaining))
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    output, _ = child.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    output, _ = child.communicate()
        log = OUT / (label + ".txt")
        with log.open("x") as stream:
            stream.write(output)
        report["processes"].append({"label": label, "exit_code": child.returncode,
            "timed_out": timeout, "wall_seconds": time.monotonic() - step, "stdout_sha256": sha(log)})
        save()
        if timeout or child.returncode:
            raise RuntimeError("owned step failed; actual stdout retained")
        return output

    def data_hashes(directory):
        return {str(path.relative_to(OUT)): sha(path)
                for path in directory.iterdir() if path.is_file()}

    code = 1
    try:
        output = run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                      "-p", "test_participant_r11.py", "-v"], "linux_new_r11_control_tests", 60)
        if not re.search(r"Ran 8 tests in", output) or not re.search(r"\nOK\s*$", output) or "skipped" in output.lower():
            raise ValueError("eight actual native checks without skips required")
        report["new_linux_tests_without_skips"] = 8
        for variant in ("r05", "r11"):
            source = ROOT / ("participant_" + variant)
            archive = OUT / (variant + ".zip")
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
                for name in PAYLOAD:
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.external_attr = 0o100644 << 16
                    bundle.writestr(entry, (source / name).read_bytes())
            if variant == "r05" and sha(archive) != "e0532e65eb5e6d35ce7463d61d101549391ef16ecefabbb9226d46aba16cbddd":
                raise ValueError("original reference ZIP differs")
            run([sys.executable, str(REFERENCE / "tools/check_submission_zip.py"), str(archive)],
                "actual_zip_gate_" + variant, 30)
            extracted = OUT / ("submission_" + variant)
            extracted.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                if set(bundle.namelist()) != set(PAYLOAD) or len(bundle.namelist()) != 3:
                    raise ValueError("unexpected ZIP payload")
                for name in PAYLOAD:
                    data = bundle.read(name)
                    if data != (source / name).read_bytes():
                        raise ValueError("actual ZIP differs from frozen source")
                    with (extracted / name).open("xb") as stream:
                        stream.write(data)
            report["archives"].append({"variant": variant, "sha256": sha(archive), "bytes": archive.stat().st_size,
                                       "status": "PASS", "actual_extracted_payloads_verified": True})
        for seed in SEEDS:
            report["candidate_parent_seconds_by_seed"][str(seed)] = 0.0
            for instrument in INSTRUMENTS:
                schema = REFERENCE / "data" / (instrument + ".json")
                dataset = OUT / f"data_{seed}_{instrument}"
                run([sys.executable, str(REFERENCE / "make_sandbox.py"), "--schema", str(schema),
                     "--config", str(REFERENCE / "config.yml"), "--seed", str(seed), "--out", str(dataset)],
                    f"generate_{seed}_{instrument}", 45)
                frozen_data = data_hashes(dataset)
                report["dataset_hashes"].update(frozen_data)
                pair_cells = {}
                for variant in ("r05", "r11"):
                    if data_hashes(dataset) != frozen_data:
                        raise ValueError("paired dataset changed")
                    limit = 300 if variant == "r05" else CAPS[instrument]
                    log = OUT / f"{seed}_{instrument}_{variant}.score.json"
                    run([sys.executable, str(REFERENCE / "score.py"), "--submission",
                         str(OUT / ("submission_" + variant)), "--data", str(dataset), "--schema", str(schema),
                         "--config", str(REFERENCE / "config.yml"), "--phase", "1", "--seed", str(seed),
                         "--timeout", str(limit), "--log", str(log)],
                        f"score_{seed}_{instrument}_{variant}", limit + 120)
                    if variant == "r11":
                        report["candidate_parent_seconds_by_seed"][str(seed)] += report["processes"][-1]["wall_seconds"]
                        if report["candidate_parent_seconds_by_seed"][str(seed)] > 900:
                            raise RuntimeError("candidate conservative shared900 parent allowance exceeded")
                    result = json.loads(log.read_text())
                    if result["status"] != "PASS" or not math.isfinite(result["skill"]):
                        raise ValueError("invalid score")
                    if data_hashes(dataset) != frozen_data:
                        raise ValueError("scoring changed the input dataset")
                    cell = {key: result[key] for key in ("skill", "reported_skill", "log_score", "n_cells",
                                                        "n_held_out", "n_respondents", "uniform_reference", "by_item")}
                    cell.update(variant=variant, seed=seed, instrument=instrument, status="PASS",
                                score_log_sha256=sha(log), actual_archive_execution=True)
                    report["cells"].append(cell)
                    pair_cells[variant] = cell
                    print("CELL_JSON " + json.dumps({key: cell[key] for key in
                          ("seed", "instrument", "variant", "skill", "n_cells")}, sort_keys=True), flush=True)
                assert all(pair_cells["r05"][key] == pair_cells["r11"][key]
                           for key in ("n_cells", "n_held_out", "n_respondents"))
                pair = {"seed": seed, "instrument": instrument,
                        "delta_skill": pair_cells["r11"]["skill"] - pair_cells["r05"]["skill"]}
                report["pairs"].append(pair)
                wins = sum(p["delta_skill"] >= .005 for p in report["pairs"])
                regressions = sum(p["delta_skill"] < -.015 for p in report["pairs"])
                print("PAIR_JSON " + json.dumps(pair, sort_keys=True), flush=True)
                if regressions or wins + 6 - len(report["pairs"]) < 4:
                    report.update(status="STOP_FUTILITY", useful_pairs=wins, material_regression_pairs=regressions,
                                  unstarted_pairs=6 - len(report["pairs"]),
                                  stop_reason="MATERIAL_REGRESSION" if regressions else "MAX_REMAINING_WINS_BELOW_REQUIRED_FOUR")
                    break
            if report["status"] == "STOP_FUTILITY":
                break
        else:
            report.update(status="PASS", useful_pairs=wins, material_regression_pairs=regressions,
                          feasibility_gate=len(report["cells"]) == 12 and wins >= 4 and regressions == 0)
        code = 0
    except Exception as error:
        report.update(status="FAIL", error_class=type(error).__name__, error=str(error))
    report["frozen_inputs_unchanged"] = all(sha(ROOT / name) == expected for name, expected in manifest["inputs"].items())
    if not report["frozen_inputs_unchanged"]:
        report["status"], code = "FAIL", 1
    report.update(wall_seconds=time.monotonic() - started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print("FINAL_RECEIPT_JSON " + json.dumps(report, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
