"""One independent Linux actual-archive contract, not a ranking experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "_reference"
OFFICIAL_COMMIT = "0d2332d8ae19a8ce171031142bdc97134910e7ec"
ARCHIVE_SHA = "a5295ed4511cc6a950778cd12b1710cdbf7273b0d93326e4a0c8c2b9e07fc4e8"
PAYLOAD = ("main.py", "requirements.txt", "LICENSE", "evidence_core.py", "latent_core.py")
INSTRUMENTS = ("unhcr", "unicef", "world_bank")
CAPS = {"unhcr": 900, "unicef": 300, "world_bank": 300}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    manifest = json.loads((ROOT / "R09_SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["inputs"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("frozen participant input differs")
    head = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"], text=True, timeout=15).strip()
    if head != OFFICIAL_COMMIT or dirty:
        raise ValueError("official reference must be exact and clean")
    return {"participant_inputs": manifest["inputs"], "source_manifest_sha256": sha(ROOT / "R09_SOURCE_MANIFEST.json"),
            "official_commit": head, "official_files": {name: sha(REFERENCE / name) for name in
            ("score.py", "make_sandbox.py", "config.yml", "LICENSE", "tools/check_submission_zip.py")}}


def worker(instrument, out, submission):
    sys.path.insert(0, str(REFERENCE))
    import numpy as np
    import pandas as pd
    import score as grader
    from make_sandbox import load_config, load_schema, scored_items
    config = load_config(str(REFERENCE / "config.yml"))
    schema = load_schema(str(REFERENCE / "data" / (instrument + ".json")), config)
    assert config["phases"][2]["timeout_seconds"] == 3600
    data = ROOT / "_native_r09" / ("data_" + instrument)
    delivered = grader.load_frames(str(data), schema)
    masked, cells, fabricated_truth = grader.sample_rows(schema, delivered, 2)
    del fabricated_truth
    targets = scored_items(schema)
    visible = int(delivered["role"].isin(("TRAIN", "DEV")).sum())
    hidden = int((delivered["role"] == "TEST").sum())
    assert len(masked) == visible + hidden and len(cells) == hidden * len(targets)
    assert masked.iloc[:visible][targets].notna().all().all()
    assert masked.iloc[visible:][targets].isna().all().all()
    with tempfile.TemporaryDirectory(prefix="r09-native-", dir=out) as directory:
        grader.stage(schema, masked, directory)
        hashes = {name: sha(Path(directory) / name) for name in ("data.json", "schema.json")}
        started = time.monotonic()
        vectors = grader.run_submission(sys.executable, str(submission), directory, CAPS[instrument],
            config["scoring"]["docker_image"], config["runner"], docker=False, verbose=True)
        wall = time.monotonic() - started
        grader.check(schema, vectors, cells)
        for vector in vectors:
            array = np.asarray(vector, dtype=float)
            assert np.isfinite(array).all() and (array > 0).all()
            assert abs(float(array.sum())-1) < 1e-8
        assert all(sha(Path(directory)/name) == expected for name, expected in hashes.items())
    contract = {"instrument": instrument, "phase": 2, "status": "PASS", "vectors_checked": len(vectors),
        "visible": visible, "hidden": hidden, "target_items": len(targets), "predict_seconds": wall,
        "positive_finite_normalized": True, "full_predict_block_hidden": True,
        "stage_inputs_unchanged": True, "official_socket_guard": True,
        "scoring_function_called": False, "numpy": np.__version__, "pandas": pd.__version__}
    with (out/"contract.json").open("x") as handle:
        json.dump(contract, handle, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=INSTRUMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("this independent contract requires native Linux x86_64 Python3.12")
    if args.worker:
        worker(args.worker, args.out, args.submission)
        return 0
    frozen = verify_sources()
    out = ROOT/"_native_r09"
    out.mkdir(mode=0o700)
    started = time.monotonic()
    report = {"scope": "INDEPENDENT_LINUX_ACTUAL_ZIP_PHASE2_CONTRACT_NOT_SCORING",
        "status": "FAIL", "source_commit": os.environ.get("GITHUB_SHA"),
        "repository": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
        "source": frozen, "contracts": [], "scoring_function_called": False,
        "official_score": None, "real_microdata_used": False, "gpu_h100_equivalence": False,
        "new_service_spend_cny": 0, "processes": []}

    def run(command, name, cap):
        remaining = 1800-(time.monotonic()-started)
        if remaining <= 0 or os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 10*1024**3:
            raise RuntimeError("native resource gate")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 50*1024**2:
            raise RuntimeError("native output exceeds50MiB")
        before = time.monotonic()
        with subprocess.Popen(command, cwd=REFERENCE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True) as child:
            try:
                output, _ = child.communicate(timeout=min(cap, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    output, _ = child.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    output, _ = child.communicate()
                report["processes"].append({"name": name, "exit_code": child.returncode,
                    "timed_out": True, "wall_seconds": time.monotonic()-before})
                print("STEP_TIMEOUT "+name+" "+output[-1200:], flush=True)
                raise TimeoutError("owned native step timeout")
        report["processes"].append({"name": name, "exit_code": child.returncode,
                                    "wall_seconds": time.monotonic()-before})
        if child.returncode:
            print("STEP_FAILED "+name+" "+output[-1200:], flush=True)
            raise RuntimeError("owned native step failed")
        return output

    try:
        test_output = run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT/"tests"),
             "-p", "test_participant_r09.py", "-v"], "new_linux_selector_tests", 45)
        if not re.search(r"Ran 8 tests in [0-9.]+s\s+OK\s*$", test_output):
            raise ValueError("native suite must actually run eight tests without skips")
        report["new_linux_selector_tests"] = 8
        archive = out/"r09.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundled:
            for name in PAYLOAD:
                entry = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                entry.external_attr = 0o100644 << 16
                bundled.writestr(entry, (ROOT/"participant_r09"/name).read_bytes())
        if sha(archive) != ARCHIVE_SHA:
            raise ValueError("exact native ZIP differs from locally selected archive")
        run([sys.executable, str(REFERENCE/"tools/check_submission_zip.py"), str(archive)], "new_native_zip_contract", 45)
        report["archive_sha256"], report["archive_bytes"] = sha(archive), archive.stat().st_size
        submission = out/"submission_from_zip"
        submission.mkdir()
        with zipfile.ZipFile(archive) as bundled:
            assert bundled.namelist() == list(PAYLOAD)
            for name in PAYLOAD:
                payload = bundled.read(name)
                assert payload == (ROOT/"participant_r09"/name).read_bytes()
                with (submission/name).open("xb") as handle:
                    handle.write(payload)
        for instrument in INSTRUMENTS:
            data = out/("data_"+instrument)
            run([sys.executable, str(REFERENCE/"make_sandbox.py"), "--schema", str(REFERENCE/"data"/(instrument+".json")),
                 "--config", str(REFERENCE/"config.yml"), "--seed", "20261002", "--out", str(data)],
                "generate_public_"+instrument, 60)
            child_out = out/(instrument+"_phase2")
            child_out.mkdir()
            run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", instrument,
                 "--out", str(child_out), "--submission", str(submission)], instrument+"_phase2", CAPS[instrument]+180)
            report["contracts"].append(json.loads((child_out/"contract.json").read_text()))
            print("NATIVE_CONTRACT "+json.dumps(report["contracts"][-1], sort_keys=True), flush=True)
        assert sum(c["predict_seconds"] for c in report["contracts"]) <= 1500
        assert sum(c["vectors_checked"] for c in report["contracts"]) == 442425
        assert verify_sources() == frozen
        report["status"] = "PASS"
    except Exception as error:
        report["error_class"] = type(error).__name__
    report["wall_seconds"] = time.monotonic()-started
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    print("NATIVE_RESULT_JSON "+json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
