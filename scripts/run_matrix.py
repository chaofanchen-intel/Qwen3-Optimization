#!/usr/bin/env python3
"""Run matched OpenVINO GenAI Python and C++ Qwen3 benchmark cases."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "benchmark_matrix.json"
DEFAULT_OV_ROOT = REPO / "benchmark" / "openvino"
DEFAULT_PATCH = DEFAULT_OV_ROOT / "qwen3_patch" / "opt1_rms_block_access.patch"
SOURCE_RELATIVE = Path("src/plugins/intel_gpu/src/kernel_selector/cl_kernels/rms_gpu_bfyx_opt.cl")
PLUGIN_RELATIVE = Path("install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=None, help="output directory name; defaults to UTC timestamp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="matrix model ids; default is every model")
    parser.add_argument("--backends", nargs="+", choices=("python", "cpp"), default=("python", "cpp"))
    parser.add_argument("--prefill-only", action="store_true", help="measure cold prefill with one generated token")
    parser.add_argument("--patch-compare", action="store_true",
                        help="measure unpatched baseline first, then patched optimized arm")
    parser.add_argument("--openvino-root", type=Path, default=None)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-latest", action="store_true", help=argparse.SUPPRESS)
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
        command = [
            env["Q3_PYTHON"],
            str(ov_genai_root / "tools" / "llm_bench" / "benchmark.py"),
            "-m", str(model_path), "-d", device, "-n", str(iterations), "-ic", str(output_tokens),
            "-pf", str(prompt_path), "-pi", prompt_index, "-t", "text_gen", "-bs", "1",
            "--resource_monitor", "0", "-mc", "0",
            "-r", str(output_dir / "detailed.csv"), "-rj", str(output_dir / "detailed.json"),
            "--benchmark_csv", str(output_dir / "summary.csv"), "--benchmark_json", str(output_dir / "summary.json"),
        ]
        return command
    command = [
        str(ROOT.parent / "model_bench" / "build" / "model_bench"), "llm",
        "-m", str(model_path), "-d", device, "-n", str(iterations), "-ic", str(output_tokens),
        "-pf", str(prompt_path), "-pi", prompt_index, "--no-monitor",
        "--cooldown-between-runs", "0", "-r", str(output_dir / "detailed.csv"), "-rj", str(output_dir / "detailed.json"),
    ]
    return command


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


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"$ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_run(ov_root: Path, arguments: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ov_root), *arguments],
        text=True,
        capture_output=True,
        check=check,
    )


def patch_check(ov_root: Path, patch: Path, reverse: bool) -> bool:
    arguments = ["apply"]
    if reverse:
        arguments.append("--reverse")
    arguments.extend(["--check", str(patch)])
    return git_run(ov_root, arguments, check=False).returncode == 0


def patch_state(ov_root: Path, patch: Path) -> str:
    forward = patch_check(ov_root, patch, reverse=False)
    reverse = patch_check(ov_root, patch, reverse=True)
    if forward and not reverse:
        return "unapplied"
    if reverse and not forward:
        return "applied"
    raise RuntimeError("cannot determine patch state; expected exactly one clean apply direction")


def set_patch_state(ov_root: Path, patch: Path, target: str) -> None:
    current = patch_state(ov_root, patch)
    if current == target:
        return
    if target == "applied":
        run_command(["git", "-C", str(ov_root), "apply", str(patch)], REPO)
    elif target == "unapplied":
        run_command(["git", "-C", str(ov_root), "apply", "--reverse", str(patch)], REPO)
    else:
        raise ValueError(f"unknown patch state: {target}")
    if patch_state(ov_root, patch) != target:
        raise RuntimeError(f"patch state did not become {target}")


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_diff_digest(ov_root: Path) -> str:
    result = git_run(ov_root, ["diff", "--binary", "--", str(SOURCE_RELATIVE)])
    return hashlib.sha256(result.stdout.encode()).hexdigest()


def patch_provenance(ov_root: Path, patch: Path, build_dir: Path) -> dict:
    source_path = ov_root / SOURCE_RELATIVE
    plugin_path = ov_root / PLUGIN_RELATIVE
    return {
        "source_commit": git_run(ov_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "patch_state": patch_state(ov_root, patch),
        "patch_file": str(patch),
        "patch_sha256": file_digest(patch),
        "source_file": str(source_path),
        "source_file_sha256": file_digest(source_path),
        "source_diff_sha256": source_diff_digest(ov_root),
        "build_dir": str(build_dir),
        "plugin_artifact": str(plugin_path),
        "plugin_artifact_sha256": file_digest(plugin_path),
    }


def build_plugin(ov_root: Path, build_dir: Path, jobs: int) -> list[str]:
    build_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "openvino_intel_gpu_plugin",
        "--parallel",
        str(jobs),
    ]
    install_command = ["cmake", "--install", str(build_dir)]
    run_command(build_command, REPO)
    run_command(install_command, REPO)
    plugin_path = ov_root / PLUGIN_RELATIVE
    if not plugin_path.is_file():
        raise RuntimeError(f"plugin artifact was not installed: {plugin_path}")
    return [command_text(build_command), command_text(install_command)]


def patch_child_environment(ov_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    install = ov_root / "install"
    library_paths = [
        install / "runtime" / "lib" / "intel64",
        install / "runtime" / "3rdparty" / "tbb" / "lib",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in library_paths] + ([existing] if existing else [])
    )
    return env


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_cpp_ttft(arm_dir: Path, model_id: str) -> dict:
    report_path = arm_dir / "raw" / "cpp" / model_id / "detailed.json"
    entries = json.loads(report_path.read_text())
    entry = entries[0]
    return {
        "mean_ms": float(entry["ttft_mean_ms"]),
        "std_ms": float(entry["ttft_std_ms"]),
    }


def render_patch_comparison(run_dir: Path, rows: list[dict], manifest: dict) -> None:
    write_json(run_dir / "comparison.json", {"manifest": manifest, "rows": rows})
    fields = [
        "id",
        "family",
        "precision",
        "int4_variant",
        "baseline_mean_ms",
        "optimized_mean_ms",
        "delta_ms",
        "delta_percent",
        "baseline_std_ms",
        "optimized_std_ms",
    ]
    with (run_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Cold-start TTFT patch comparison: `{run_dir.name}`",
        "",
        "Baseline is measured before applying the source patch. Optimized is measured after applying it.",
        "Both arms use prompt permutation ON and `-ic 1`; values below are arithmetic means over the measured iterations.",
        "",
        "| Model | Variant | Baseline mean ms | Optimized mean ms | Delta ms | Delta | Baseline std | Optimized std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        variant = row["precision"]
        if row["int4_variant"]:
            variant += f" {row['int4_variant']}"
        lines.append(
            f"| {row['family']} | {variant} | {row['baseline_mean_ms']:.2f} | "
            f"{row['optimized_mean_ms']:.2f} | {row['delta_ms']:+.2f} | "
            f"{row['delta_percent']:+.2f}% | {row['baseline_std_ms']:.2f} | "
            f"{row['optimized_std_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The complete provenance and raw logs are in `baseline/`, `optimized/`, and `metadata.json`.",
        ]
    )
    (run_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def run_patch_arm(
    arm: str,
    arm_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    model_entries: list[dict],
    ov_root: Path,
    patch: Path,
    build_dir: Path,
    manifest: dict,
    manifest_path: Path,
) -> dict:
    expected_state = "unapplied" if arm == "baseline" else "applied"
    if patch_state(ov_root, patch) != expected_state:
        raise RuntimeError(f"{arm} arm started with the wrong patch state")

    build_commands = build_plugin(ov_root, build_dir, args.jobs)
    provenance = patch_provenance(ov_root, patch, build_dir)
    provenance["build_commands"] = build_commands
    manifest["arms"].append({"name": arm, "status": "built", "provenance": provenance})
    write_json(manifest_path, manifest)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_matrix.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(arm_dir),
        "--no-latest",
        "--prefill-only",
        "--backends",
        "cpp",
        "--iterations",
        str(args.iterations),
    ]
    if args.models:
        command.extend(["--models", *args.models])
    run_command(command, ROOT, patch_child_environment(ov_root))

    results = {
        model["id"]: read_cpp_ttft(arm_dir, model["id"])
        for model in model_entries
    }
    manifest["arms"][-1]["status"] = "completed"
    write_json(manifest_path, manifest)
    return {"provenance": provenance, "results": results, "command": command_text(command)}


def run_patch_comparison(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    model_entries = config["models"]
    if args.models:
        wanted = set(args.models)
        model_entries = [entry for entry in model_entries if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in model_entries}
        if missing:
            raise SystemExit(f"unknown model ids: {', '.join(sorted(missing))}")

    ov_root = (args.openvino_root or Path(os.environ.get("Q3_OV_ROOT", DEFAULT_OV_ROOT))).resolve()
    patch = args.patch.resolve()
    build_dir = ov_root / "build"
    run_id = args.run_id or time.strftime("cold_ttft_patch_compare_%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / run_id
    if run_dir.exists():
        raise SystemExit(f"result directory already exists: {run_dir}")
    if not patch.is_file():
        raise SystemExit(f"patch file not found: {patch}")
    if not build_dir.is_dir():
        raise SystemExit(f"OpenVINO build directory not found: {build_dir}")

    initial_state = patch_state(ov_root, patch)
    manifest = {
        "run_id": run_id,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "openvino_root": str(ov_root),
        "openvino_commit": git_run(ov_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "patch": str(patch),
        "initial_patch_state": initial_state,
        "protocol": {
            "input_tokens": config["protocol"]["input_tokens"],
            "output_tokens": 1,
            "iterations": args.iterations,
            "warmup_iterations": 1,
            "prompt_permutation": True,
            "summary_statistic": "arithmetic_mean",
            "device": config["protocol"]["device"],
            "attention_backend": "PA (ov::genai::LLMPipeline default)",
        },
        "models": [entry["id"] for entry in model_entries],
        "arms": [],
    }
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "metadata.json"
    write_json(manifest_path, manifest)

    try:
        set_patch_state(ov_root, patch, "unapplied")
        baseline_dir = run_dir / "baseline"
        baseline = run_patch_arm(
            "baseline", baseline_dir, args, config_path, model_entries, ov_root,
            patch, build_dir, manifest, manifest_path,
        )
        set_patch_state(ov_root, patch, "applied")
        optimized_dir = run_dir / "optimized"
        optimized = run_patch_arm(
            "optimized", optimized_dir, args, config_path, model_entries, ov_root,
            patch, build_dir, manifest, manifest_path,
        )

        rows = []
        for model in model_entries:
            baseline_stats = baseline["results"][model["id"]]
            optimized_stats = optimized["results"][model["id"]]
            delta_ms = optimized_stats["mean_ms"] - baseline_stats["mean_ms"]
            rows.append(
                {
                    "id": model["id"],
                    "family": model["family"],
                    "precision": model["precision"],
                    "int4_variant": model.get("int4_variant"),
                    "baseline_mean_ms": baseline_stats["mean_ms"],
                    "optimized_mean_ms": optimized_stats["mean_ms"],
                    "delta_ms": delta_ms,
                    "delta_percent": delta_ms / baseline_stats["mean_ms"] * 100.0,
                    "baseline_std_ms": baseline_stats["std_ms"],
                    "optimized_std_ms": optimized_stats["std_ms"],
                }
            )
        render_patch_comparison(run_dir, rows, manifest)
        manifest["status"] = "completed"
        manifest["final_patch_state"] = patch_state(ov_root, patch)
        write_json(manifest_path, manifest)
        print(f"comparison: {run_dir / 'comparison.md'}")
        return 0
    finally:
        set_patch_state(ov_root, patch, initial_state)
        manifest["final_patch_state"] = patch_state(ov_root, patch)
        if manifest.get("status") != "completed":
            manifest["status"] = "failed"
        write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    if args.patch_compare:
        return run_patch_comparison(args)

    config = json.loads(args.config.read_text())
    protocol = dict(config["protocol"])
    if args.device:
        protocol["device"] = args.device
    if args.iterations is not None:
        protocol["iterations"] = args.iterations
    if args.output_tokens is not None:
        protocol["output_tokens"] = args.output_tokens
    protocol["prompt_permutation"] = True
    protocol["summary_statistic"] = "arithmetic_mean"
    if args.prefill_only:
        protocol["output_tokens"] = 1
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
    run_dir = args.output_dir.resolve() if args.output_dir else ROOT / "results" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run directory already exists and is not empty: {run_dir}")
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
    if not args.no_latest:
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