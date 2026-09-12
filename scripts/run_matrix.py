#!/usr/bin/env python3
"""Run matched OpenVINO GenAI Python and C++ Qwen3 benchmark cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "benchmark_matrix.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=None, help="output directory name; defaults to UTC timestamp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="matrix model ids; default is every model")
    parser.add_argument("--backends", nargs="+", choices=("python", "cpp"), default=("python", "cpp"))
    parser.add_argument("--smoke", action="store_true", help="one measured iteration and eight output tokens")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_environment() -> dict[str, str]:
    q3_python_env = Path(os.environ.get("Q3_PYTHON_ENV", Path.home() / "envs" / "qwen_env"))
    env = os.environ.copy()
    env["Q3_PYTHON_ENV"] = str(q3_python_env)
    env["Q3_PYTHON"] = str(q3_python_env / "bin" / "python")
    env["Q3_OPTIMUM_CLI"] = str(q3_python_env / "bin" / "optimum-cli")
    ov_root = Path(os.environ.get("Q3_OV_ROOT", ROOT.parent / "benchmark" / "openvino"))
    ov_install = ov_root / "install"
    ov_genai_root = Path(str(ov_root) + ".genai")
    env["Q3_OV_ROOT"] = str(ov_root)
    env["Q3_OV_INSTALL"] = str(ov_install)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ov_genai_root / "tools" / "llm_bench"), str(ov_install / "python"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(ov_install / "runtime" / "lib" / "intel64"), str(ov_install / "runtime" / "3rdparty" / "tbb" / "lib"), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return env


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def command_for(backend: str, model_path: Path, prompt_path: Path, output_dir: Path, protocol: dict, env: dict[str, str]) -> list[str]:
    device = protocol["device"]
    iterations = protocol["iterations"]
    output_tokens = protocol["output_tokens"]
    prompt_index = str(protocol.get("prompt_index", 0))
    if backend == "python":
        ov_genai_root = Path(env["Q3_OV_ROOT"] + ".genai")
        return [
            env["Q3_PYTHON"],
            str(ov_genai_root / "tools" / "llm_bench" / "benchmark.py"),
            "-m", str(model_path), "-d", device, "-n", str(iterations), "-ic", str(output_tokens),
            "-pf", str(prompt_path), "-pi", prompt_index, "-t", "text_gen", "-bs", "1",
            "--disable_prompt_permutation", "--resource_monitor", "0", "-mc", "0",
            "-r", str(output_dir / "detailed.csv"), "-rj", str(output_dir / "detailed.json"),
            "--benchmark_csv", str(output_dir / "summary.csv"), "--benchmark_json", str(output_dir / "summary.json"),
        ]
    return [
        str(ROOT.parent / "model_bench" / "build" / "model_bench"), "llm",
        "-m", str(model_path), "-d", device, "-n", str(iterations), "-ic", str(output_tokens),
        "-pf", str(prompt_path), "-pi", prompt_index, "--disable-prompt-permutation", "--no-monitor",
        "--cooldown-between-runs", "0", "-r", str(output_dir / "detailed.csv"), "-rj", str(output_dir / "detailed.json"),
    ]


def capture_versions(env: dict[str, str]) -> dict[str, str]:
    code = (
        "import openvino, openvino_genai, importlib.metadata as m; "
        "print('openvino=' + openvino.__version__); "
        "print('openvino_genai=' + openvino_genai.__version__); "
        "print('optimum_intel=' + m.version('optimum-intel')); "
        "print('nncf=' + m.version('nncf')); "
        "print('transformers=' + m.version('transformers')); "
        "print('torch=' + m.version('torch'))"
    )
    result = subprocess.run([env["Q3_PYTHON"], "-c", code], env=env, text=True, capture_output=True, check=False)
    versions: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            versions[key] = value
    return versions


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    protocol = dict(config["protocol"])
    if args.device:
        protocol["device"] = args.device
    if args.iterations is not None:
        protocol["iterations"] = args.iterations
    if args.output_tokens is not None:
        protocol["output_tokens"] = args.output_tokens
    if args.smoke:
        protocol["iterations"] = 1
        protocol["output_tokens"] = 8

    model_entries = config["models"]
    if args.models:
        wanted = set(args.models)
        model_entries = [entry for entry in model_entries if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in model_entries}
        if missing:
            raise SystemExit(f"unknown model ids: {', '.join(sorted(missing))}")

    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("gpu_1k_512_%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}; choose another --run-id")
    raw_dir = run_dir / "raw"
    env = load_environment()
    prompt_path = resolve_path(ROOT, protocol["prompt_file"])
    if not prompt_path.is_file():
        raise SystemExit(f"prompt file not found: {prompt_path}")
    if not Path(env["Q3_PYTHON"]).is_file():
        raise SystemExit(f"requested Python environment not found: {env['Q3_PYTHON']}")

    metadata = {
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(ROOT),
        "protocol": protocol,
        "prompt_file": str(prompt_path),
        "backends": list(args.backends),
        "python_executable": env["Q3_PYTHON"],
        "versions": capture_versions(env),
        "host": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "models": {entry["id"]: entry for entry in model_entries},
        "commands": [],
        "failures": [],
    }
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    failures = 0
    for entry in model_entries:
        model_path = resolve_path(ROOT, entry["path"])
        if not model_path.is_dir():
            message = f"model directory not found: {model_path}"
            metadata["failures"].append({"model": entry["id"], "error": message})
            print(f"[missing] {entry['id']}: {message}")
            failures += 1
            continue
        for backend in args.backends:
            case_dir = raw_dir / backend / entry["id"]
            case_dir.mkdir(parents=True)
            command = command_for(backend, model_path, prompt_path, case_dir, protocol, env)
            log_path = case_dir / "run.log"
            metadata["commands"].append({"backend": backend, "model": entry["id"], "argv": command, "shell": shlex.join(command)})
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"[run] {backend:6s} {entry['id']} -> {case_dir}")
            with log_path.open("w") as log:
                log.write("$ " + shlex.join(command) + "\n\n")
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures += 1
                message = {"backend": backend, "model": entry["id"], "returncode": result.returncode, "log": str(log_path)}
                metadata["failures"].append(message)
                print(f"[failed] {backend} {entry['id']} returncode={result.returncode}; see {log_path}")
                if not args.continue_on_error:
                    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    return 1
            else:
                print(f"[done] {backend:6s} {entry['id']}")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    latest = ROOT / "results" / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    print(f"run directory: {run_dir}")
    if failures:
        print(f"completed with {failures} failure(s); use --continue-on-error to collect the rest")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())