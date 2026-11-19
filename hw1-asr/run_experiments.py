"""Reproduce task-guide parameter sweeps without changing submitted sources."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "experiments"
ROW = re.compile(r"^(.*?)\s+(PASS|FAIL)\s+max_err=(\S+)(?:\s+(\S+)\s+ms\s+(.*))?$")


def parse(text):
    cases = []
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if match:
            name, status, error, ms, speed = match.groups()
            row = {"case": name.strip(), "status": status, "max_error": float(error)}
            if ms is not None:
                row["ms"] = float(ms)
                throughput = re.search(r"([\d.]+)\s+(GB/s|GFLOP/s)", speed)
                if throughput:
                    row["throughput"] = float(throughput[1])
                    row["throughput_unit"] = throughput[2]
            cases.append(row)
    return cases


def run(args, log):
    result = subprocess.run(args, cwd=HERE, capture_output=True, text=True, timeout=600)
    text = result.stdout + result.stderr
    log.write_text(text)
    if result.returncode or "ALL PASS" not in text:
        raise RuntimeError(f"Failed: {args}; see {log}")
    return parse(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sm = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).splitlines()[0].strip().replace(".", "")
    metadata = {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "gpu": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        "started_unix": time.time(), "repeats": 3, "variants": [],
    }
    tasks = [
        (1, "task1_rope.cu", "ROPE_THREADS", 256, [32, 64, 128, 512, 1024], "rope"),
        (2, "task2_reduce.cu", "REDUCE_THREADS", 256, [32, 64, 128, 512, 1024], "reduce"),
        (3, "task3_gemm.cu", "TILE", 16, [8, 32], "gemm"),
        (4, "task4_attention.cu", "ATTN_WARPS", 4, [1, 2, 8, 16], "attention"),
    ]
    for task, filename, parameter, default, values, suffix in tasks:
        for value in [default] + values:
            name = f"task{task}_{parameter.lower()}_{value}"
            print(name, flush=True)
            repeats = []
            if value == default:
                binary = HERE / f"build/sm_{sm}/test_task{task}_{suffix}"
                first = 0
            else:
                source = OUT / "src" / name
                shutil.copytree(HERE / "csrc", source, dirs_exist_ok=True)
                file = source / filename
                text, count = re.subn(
                    rf"constexpr int {parameter} = {default};",
                    f"constexpr int {parameter} = {value};", file.read_text(),
                )
                if count != 1:
                    raise RuntimeError("Unexpected tuning parameter declaration")
                file.write_text(text)
                build = OUT / "build" / name
                repeats.append(run(
                    ["make", "-j2", f"SRC={source}", f"BUILD={build}", f"test{task}"],
                    OUT / f"{name}_run1.log",
                ))
                binary = build / f"test_task{task}_{suffix}"
                first = 1
            for repeat in range(first, 3):
                repeats.append(run([str(binary)], OUT / f"{name}_run{repeat+1}.log"))
            metadata["variants"].append({
                "task": task, "parameter": parameter, "value": value,
                "default": default, "repeats": repeats,
            })
            (OUT / "experiments.json").write_text(json.dumps(metadata, indent=2))

    for task, suffix in [(1, "rope"), (2, "reduce"), (3, "gemm"), (4, "attention")]:
        tool = "memcheck" if task == 1 else "racecheck"
        command = ["compute-sanitizer", "--error-exitcode=99", "--tool", tool,
                   str(HERE / f"build/sm_{sm}/test_task{task}_{suffix}"), "small"]
        result = subprocess.run(command, cwd=HERE, capture_output=True, text=True, timeout=300)
        (OUT / f"sanitizer_task{task}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Sanitizer failed for task {task}")
    metadata["finished_unix"] = time.time()
    metadata["sanitizers"] = "All four small-case sanitizer runs passed"
    (OUT / "experiments.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
