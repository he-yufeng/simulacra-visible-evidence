"""One bounded native cohort: exact R11/R12 ZIPs, invented data, no private API."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "_reference"
COMMIT = "0d2332d8ae19a8ce171031142bdc97134910e7ec"
HASHES = {"r11": "32f16501feeb7b6890262165d3f510b14b818208e0ee73aca716d847ead6cdf8",
          "r12": "aa24868f9a048316a904447f26383db6beca65ccba584eac453817ba24e7f1a6"}
SEEDS = {1: 202610131, 2: 202610132}
INSTRUMENTS = ("unhcr", "unicef", "world_bank")
CAPS = {"r11": {"unhcr": 480, "unicef": 120, "world_bank": 240},
        "r12": {"unhcr": 600, "unicef": 90, "world_bank": 150}}
PAYLOAD = ("main.py", "requirements.txt", "LICENSE")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(pairs):
    if len(pairs) > 6 or any(not math.isfinite(p["delta_skill"]) for p in pairs):
        raise ValueError("invalid comparison list")
    wins = sum(p["delta_skill"] >= .005 for p in pairs)
    regressions = sum(p["delta_skill"] < -.015 for p in pairs)
    remaining = 6 - len(pairs)
    reason = "MATERIAL_REGRESSION" if regressions else (
        "MAX_REMAINING_WINS_BELOW_REQUIRED_FOUR" if wins + remaining < 4 else None)
    return {"useful_pairs": wins, "material_regression_pairs": regressions,
            "unstarted_pairs": remaining, "stop_reason": reason,
            "feasibility_gate": len(pairs) == 6 and wins >= 4 and regressions == 0}


def check_capacity():
    trees = []
    for variant, expected in HASHES.items():
        path = ROOT / f"participant_{variant}/main.py"
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("frozen model source mismatch")
        tree = ast.parse(path.read_text())
        values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "iterations":
                values.append(ast.literal_eval(node.value))
                node.value = ast.Constant(value=0)
        if values != [96 if variant == "r11" else 192]:
            raise ValueError("wrong fixed capacity")
        trees.append(ast.dump(tree, include_attributes=False))
    if trees[0] != trees[1]:
        raise ValueError("capacity is not the only code change")
    for name in PAYLOAD[1:]:
        if (ROOT / "participant_r11" / name).read_bytes() != (ROOT / "participant_r12" / name).read_bytes():
            raise ValueError("dependency/license changed")


def main():
    cohort = int(os.environ["R12_COHORT"])
    if cohort not in SEEDS:
        raise ValueError("unexpected cohort")
    seed, out = SEEDS[cohort], ROOT / f"_method_r12_c{cohort}"
    out.mkdir(mode=0o700)
    started = time.monotonic()
    report = {"scope": "NATIVE_EXACT_ZIP_FRESH_PUBLIC_SYNTHETIC_COHORT_NOT_OFFICIAL_SCORE",
        "status": "RUNNING", "cohort": cohort, "seed": seed,
        "source_commit": os.environ.get("GITHUB_SHA"), "repository": os.environ.get("GITHUB_REPOSITORY"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "official_commit": COMMIT,
        "cells": [], "pairs": [], "previous_pairs": [], "processes": [], "archives": [],
        "dataset_hashes": {}, "parent_seconds": {"r11": 0.0, "r12": 0.0},
        "feasibility_gate": False, "official_score": None, "official_upload_performed": False,
        "private_microdata_used": False, "gpu_h100_equivalence": False,
        "new_service_spend_cny": 0, "old_tests_or_dependency_admission_rerun": False}

    def save():
        (out / "receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2))

    def resources():
        remaining = 1800 - (time.monotonic() - started)
        disk = os.statvfs(ROOT)
        if remaining <= 0 or disk.f_bavail * disk.f_frsize < 10 * 1024**3:
            raise RuntimeError("cohort time/disk gate")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 50 * 1024**2:
            raise RuntimeError("cohort output exceeds50MiB")
        return remaining

    def run(command, label, cap):
        remaining, step = resources(), time.monotonic()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PIP_NO_CACHE_DIR="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            MPLCONFIGDIR=str(out / "mpl_config"))
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
        log = out / (label + ".txt")
        with log.open("x") as stream:
            stream.write(output)
        report["processes"].append({"label": label, "exit_code": child.returncode,
            "timed_out": timeout, "wall_seconds": time.monotonic() - step, "stdout_sha256": sha(log)})
        save()
        if timeout or child.returncode:
            raise RuntimeError("owned step failed; original output retained")
        resources()

    def hashes(directory):
        return {str(p.relative_to(out)): sha(p) for p in directory.iterdir() if p.is_file()}

    code, manifest = 1, None
    try:
        if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("native Linux x86_64 Python3.12 required")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if event["repository"]["private"] or os.environ.get("GITHUB_REPOSITORY") != "he-yufeng/simulacra-visible-evidence":
            raise ValueError("public owned repository resource contract required")
        manifest = json.loads((ROOT / "R12_SOURCE_MANIFEST.json").read_text())
        report["inputs"] = manifest["inputs"]
        for name, expected in manifest["inputs"].items():
            path = ROOT / name
            if path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError("frozen participant manifest mismatch")
        check_capacity()
        head = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"], text=True).strip()
        if head != COMMIT or dirty:
            raise ValueError("official checkout must be exact and clean")
        report["official_files"] = {name: sha(REFERENCE / name) for name in
            ("score.py", "make_sandbox.py", "config.yml", "LICENSE")}
        report["runtime"] = {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "versions": {name: metadata.version(name) for name in
            ("numpy", "pandas", "catboost", "scipy", "pyarrow", "PyYAML")}}
        report["public_standard_runner_contract"] = True
        if cohort == 2:
            previous = json.loads((ROOT / "R12_COHORT1_SUMMARY.json").read_text())
            if previous["status"] != "COHORT_COMPLETE" or previous["seed"] != SEEDS[1] or previous["model_hashes"] != HASHES:
                raise ValueError("invalid first-cohort provenance")
            report["previous_pairs"] = previous["pairs"]
            report["previous_receipt_sha256"] = previous["receipt_sha256"]
            if len(previous["pairs"]) != 3 or {p["instrument"] for p in previous["pairs"]} != set(INSTRUMENTS):
                raise ValueError("first-cohort comparison shape mismatch")
            if decision(previous["pairs"])["stop_reason"]:
                raise ValueError("closed first cohort must not be reopened")
        for variant in ("r11", "r12"):
            source, archive = ROOT / f"participant_{variant}", out / f"{variant}.zip"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
                for name in PAYLOAD:
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.external_attr = 0o100644 << 16
                    bundle.writestr(entry, (source / name).read_bytes())
            if variant == "r11" and sha(archive) != "a53dbe25f96a6ec5e14ab815c8a0bd7f4ff5a1f21174f5646595d47c94326927":
                raise ValueError("frozen R11 archive differs")
            if variant == "r12" and cohort == 1:
                run([sys.executable, str(REFERENCE / "tools/check_submission_zip.py"), str(archive)],
                    "new_r12_actual_zip_gate", 30)
            extracted = out / f"submission_{variant}"
            extracted.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                if set(bundle.namelist()) != set(PAYLOAD) or len(bundle.namelist()) != 3:
                    raise ValueError("unexpected archive payload")
                for name in PAYLOAD:
                    value = bundle.read(name)
                    if value != (source / name).read_bytes():
                        raise ValueError("archive source differs")
                    with (extracted / name).open("xb") as stream:
                        stream.write(value)
            report["archives"].append({"variant": variant, "bytes": archive.stat().st_size,
                "sha256": sha(archive), "actual_extracted_payloads_verified": True})
        for instrument in INSTRUMENTS:
            schema, dataset = REFERENCE / "data" / f"{instrument}.json", out / f"data_{seed}_{instrument}"
            run([sys.executable, str(REFERENCE / "make_sandbox.py"), "--schema", str(schema),
                 "--config", str(REFERENCE / "config.yml"), "--seed", str(seed), "--out", str(dataset)],
                f"generate_{seed}_{instrument}", 45)
            original = hashes(dataset)
            report["dataset_hashes"].update(original)
            pair_cells = {}
            for variant in ("r11", "r12"):
                if hashes(dataset) != original:
                    raise ValueError("paired input changed")
                cap, log = CAPS[variant][instrument], out / f"{seed}_{instrument}_{variant}.score.json"
                run([sys.executable, str(REFERENCE / "score.py"), "--submission", str(out / f"submission_{variant}"),
                     "--data", str(dataset), "--schema", str(schema), "--config", str(REFERENCE / "config.yml"),
                     "--phase", "1", "--seed", str(seed), "--timeout", str(cap), "--log", str(log)],
                    f"score_{seed}_{instrument}_{variant}", cap + 120)
                report["parent_seconds"][variant] += report["processes"][-1]["wall_seconds"]
                if report["parent_seconds"][variant] > 900:
                    raise RuntimeError("conservative shared900 parent allowance exceeded")
                result = json.loads(log.read_text())
                if result["status"] != "PASS" or not math.isfinite(result["skill"]) or hashes(dataset) != original:
                    raise ValueError("score or immutable data contract failed")
                cell = {key: result[key] for key in ("skill", "reported_skill", "log_score", "n_cells",
                    "n_held_out", "n_respondents", "uniform_reference", "by_item")}
                cell.update(variant=variant, seed=seed, instrument=instrument, status="PASS",
                    score_log_sha256=sha(log), actual_archive_execution=True)
                report["cells"].append(cell)
                pair_cells[variant] = cell
                print("CELL_JSON " + json.dumps({key: cell[key] for key in ("seed", "instrument", "variant", "skill", "n_cells")}), flush=True)
            if any(pair_cells["r11"][k] != pair_cells["r12"][k] for k in ("n_cells", "n_held_out", "n_respondents")):
                raise ValueError("comparison count differs")
            pair = {"seed": seed, "instrument": instrument,
                    "delta_skill": pair_cells["r12"]["skill"] - pair_cells["r11"]["skill"]}
            report["pairs"].append(pair)
            print("PAIR_JSON " + json.dumps(pair), flush=True)
            verdict = decision(report["previous_pairs"] + report["pairs"])
            report.update(verdict)
            if verdict["stop_reason"]:
                report["status"] = "STOP_FUTILITY"
                break
        else:
            report["status"] = "PASS" if report["feasibility_gate"] else "COHORT_COMPLETE"
        code = 0
    except Exception as error:
        report.update(status="FAIL", error_class=type(error).__name__, error=str(error))
    report["frozen_inputs_unchanged"] = manifest is not None and all(
        sha(ROOT / name) == value for name, value in manifest["inputs"].items())
    if not report["frozen_inputs_unchanged"]:
        report["status"], code = "FAIL", 1
    report.update(wall_seconds=time.monotonic() - started, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print("FINAL_RECEIPT_JSON " + json.dumps(report, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
