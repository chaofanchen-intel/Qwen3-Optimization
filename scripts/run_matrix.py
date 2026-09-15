#!/usr/bin/env python3
"""Run the unified Qwen3 benchmark and source-patch-stack comparison."""

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
DEFAULT_PATCHES = [
    DEFAULT_OV_ROOT / "qwen3_patch" / "opt1_rms_block_access.patch",
    DEFAULT_OV_ROOT / "qwen3_patch" / "opt2_dynamic_quantize_reuse.patch",
    DEFAULT_OV_ROOT / "qwen3_patch" / "opt3_fuse_rope_into_rms.patch",
]
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
    parser.add_argument("--patches", type=Path, nargs="+", default=DEFAULT_PATCHES,
                        help="patch stack for --patch-compare, applied in listed order")
    parser.add_argument("--patch-stage", type=int, choices=(1, 2, 3), default=None,
                        help="compare the previous stack stage with this patch stage")
    parser.add_argument("--runtime-config", type=Path, default=None,
                        help="JSON of extra OpenVINO plugin properties for C++ runs")
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
    runtime_config = protocol.get("runtime_config")
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
    if runtime_config:
        command.extend(["-lc", str(runtime_config)])
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


def patch_stack_state(ov_root: Path, patches: list[Path]) -> str:
    if not patches:
        raise RuntimeError("patch stack cannot be empty")
    first_forward = patch_check(ov_root, patches[0], reverse=False)
    last_reverse = patch_check(ov_root, patches[-1], reverse=True)
    if first_forward and not last_reverse:
        return "unapplied"
    if last_reverse and not first_forward:
        return "applied"
    raise RuntimeError(
        "cannot determine patch stack state; expected the complete stack to be "
        "cleanly applied or cleanly unapplied"
    )


def patch_prefix_state(ov_root: Path, patches: list[Path]) -> int:
    for prefix_length in range(len(patches) + 1):
        previous_applied = prefix_length == 0 or patch_check(
            ov_root, patches[prefix_length - 1], reverse=True
        )
        next_unapplied = prefix_length == len(patches) or patch_check(
            ov_root, patches[prefix_length], reverse=False
        )
        if previous_applied and next_unapplied:
            return prefix_length
    raise RuntimeError("cannot determine patch prefix state")


def set_patch_prefix(ov_root: Path, patches: list[Path], target_prefix: int) -> None:
    if not 0 <= target_prefix <= len(patches):
        raise ValueError(f"invalid patch prefix: {target_prefix}")
    current_prefix = patch_prefix_state(ov_root, patches)
    if current_prefix == target_prefix:
        return
    if current_prefix < target_prefix:
        for patch in patches[current_prefix:target_prefix]:
            run_command(["git", "-C", str(ov_root), "apply", str(patch)], REPO)
    else:
        for patch in reversed(patches[target_prefix:current_prefix]):
            run_command(["git", "-C", str(ov_root), "apply", "--reverse", str(patch)], REPO)
    if patch_prefix_state(ov_root, patches) != target_prefix:
        raise RuntimeError(f"patch prefix did not become {target_prefix}")


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_diff_digest(ov_root: Path) -> str:
    result = git_run(ov_root, ["diff", "--binary"])
    return hashlib.sha256(result.stdout.encode()).hexdigest()


def patch_provenance(
    ov_root: Path, patches: list[Path], applied_prefix: int, build_dir: Path
) -> dict:
    plugin_path = ov_root / PLUGIN_RELATIVE
    return {
        "source_commit": git_run(ov_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "patch_state": (
            "unapplied" if applied_prefix == 0
            else "applied" if applied_prefix == len(patches)
            else "partial"
        ),
        "applied_prefix": applied_prefix,
        "patches": [
            {"file": str(patch), "sha256": file_digest(patch)}
            for patch in patches
        ],
        "source_diff_sha256": source_diff_digest(ov_root),
        "source_diff_files": git_run(ov_root, ["diff", "--name-only"]).stdout.splitlines(),
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
        "Baseline is measured before applying the source patch stack. Optimized is measured after applying it.",
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
    patches: list[Path],
    applied_prefix: int,
    build_dir: Path,
    manifest: dict,
    manifest_path: Path,
) -> dict:
    if patch_prefix_state(ov_root, patches) != applied_prefix:
        raise RuntimeError(f"{arm} arm started with the wrong patch state")

    build_commands = build_plugin(ov_root, build_dir, args.jobs)
    provenance = patch_provenance(ov_root, patches, applied_prefix, build_dir)
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
    if args.runtime_config:
        command.extend(["--runtime-config", str(args.runtime_config)])
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
    patches = [patch.resolve() for patch in args.patches]
    if args.runtime_config:
        args.runtime_config = args.runtime_config.resolve()
    build_dir = ov_root / "build"
    run_id = args.run_id or time.strftime("cold_ttft_patch_compare_%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / run_id
    if run_dir.exists():
        raise SystemExit(f"result directory already exists: {run_dir}")
    missing_patches = [patch for patch in patches if not patch.is_file()]
    if missing_patches:
        raise SystemExit(f"patch file not found: {missing_patches[0]}")
    if args.runtime_config and not args.runtime_config.is_file():
        raise SystemExit(f"runtime config file not found: {args.runtime_config}")
    if not build_dir.is_dir():
        raise SystemExit(f"OpenVINO build directory not found: {build_dir}")
    if len(patches) != len(DEFAULT_PATCHES):
        raise SystemExit("--patches must name the complete three-patch stack")

    initial_prefix = patch_prefix_state(ov_root, patches)
    if args.patch_stage is None:
        baseline_prefix = 0
        optimized_prefix = len(patches)
    else:
        baseline_prefix = args.patch_stage - 1
        optimized_prefix = args.patch_stage
    manifest = {
        "run_id": run_id,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "openvino_root": str(ov_root),
        "openvino_commit": git_run(ov_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "patches": [str(patch) for patch in patches],
        "initial_patch_state": (
            "unapplied" if initial_prefix == 0
            else "applied" if initial_prefix == len(patches)
            else "partial"
        ),
        "initial_patch_prefix": initial_prefix,
        "baseline_patch_prefix": baseline_prefix,
        "optimized_patch_prefix": optimized_prefix,
        "protocol": {
            "input_tokens": config["protocol"]["input_tokens"],
            "output_tokens": 1,
            "iterations": args.iterations,
            "warmup_iterations": 1,
            "prompt_permutation": True,
            "summary_statistic": "arithmetic_mean",
            "device": config["protocol"]["device"],
            "attention_backend": "PA (ov::genai::LLMPipeline default)",
            "runtime_config": str(args.runtime_config) if args.runtime_config else None,
        },
        "models": [entry["id"] for entry in model_entries],
        "arms": [],
    }
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "metadata.json"
    write_json(manifest_path, manifest)

    try:
        set_patch_prefix(ov_root, patches, baseline_prefix)
        baseline_dir = run_dir / "baseline"
        baseline = run_patch_arm(
            "baseline", baseline_dir, args, config_path, model_entries, ov_root,
            patches, baseline_prefix, build_dir, manifest, manifest_path,
        )
        set_patch_prefix(ov_root, patches, optimized_prefix)
        optimized_dir = run_dir / "optimized"
        optimized = run_patch_arm(
            "optimized", optimized_dir, args, config_path, model_entries, ov_root,
            patches, optimized_prefix, build_dir, manifest, manifest_path,
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
        final_prefix = patch_prefix_state(ov_root, patches)
        manifest["final_patch_state"] = (
            "unapplied" if final_prefix == 0
            else "applied" if final_prefix == len(patches)
            else "partial"
        )
        manifest["final_patch_prefix"] = final_prefix
        write_json(manifest_path, manifest)
        print(f"comparison: {run_dir / 'comparison.md'}")
        return 0
    finally:
        set_patch_prefix(ov_root, patches, initial_prefix)
        manifest["final_patch_state"] = (
            "unapplied" if initial_prefix == 0
            else "applied" if initial_prefix == len(patches)
            else "partial"
        )
        manifest["final_patch_prefix"] = initial_prefix
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
    if args.runtime_config:
        if "python" in args.backends:
            raise SystemExit("--runtime-config is supported only with --backends cpp")
        runtime_config = args.runtime_config.resolve()
        if not runtime_config.is_file():
            raise SystemExit(f"runtime config file not found: {runtime_config}")
        protocol["runtime_config"] = str(runtime_config)
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