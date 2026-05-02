#!/usr/bin/env python

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", "/work/mingze/miniconda3")).resolve()
GROOT_CONDA_ENV = os.environ.get("GROOT_CONDA_ENV", os.environ.get("QUANTVLA_CONDA_ENV", "groot_test"))
LIBERO_CONDA_ENV = os.environ.get("LIBERO_CONDA_ENV", "libero_test")
GROOT_PY = os.environ.get("GROOT_PY", str(CONDA_ROOT / "envs" / GROOT_CONDA_ENV / "bin" / "python"))
LIBERO_PY = os.environ.get("LIBERO_PY", str(CONDA_ROOT / "envs" / LIBERO_CONDA_ENV / "bin" / "python"))
QUANTVLA_ROOT = Path(os.environ.get("QUANTVLA_ROOT", "/work/mingze/QuantVLA")).resolve()
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", "/work/mingze/LIBERO")).resolve()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def suite_defaults(task_suite: str) -> tuple[str, str, int]:
    if task_suite == "libero_spatial":
        return (
            "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    if task_suite == "libero_goal":
        return (
            "youliangtan/gr00t-n1.5-libero-goal-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
            10,
        )
    if task_suite == "libero_object":
        return (
            "youliangtan/gr00t-n1.5-libero-object-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    if task_suite == "libero_90":
        return (
            "youliangtan/gr00t-n1.5-libero-90-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            90,
        )
    if task_suite == "libero_10":
        return (
            "/work/mingze/checkpoints/gr00t-n1.5-libero-long-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    raise ValueError(f"Unknown TASK_SUITE: {task_suite}")


def wait_for_port(port: int, timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for localhost:{port}")


def shard_tasks(num_tasks: int, num_shards: int) -> list[list[int]]:
    shards = [[] for _ in range(num_shards)]
    for idx, task_id in enumerate(range(num_tasks)):
        shards[idx % num_shards].append(task_id)
    return shards


def make_base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("VIRTUAL_ENV", None)

    pythonpath_entries = [str(QUANTVLA_ROOT)]
    if LIBERO_ROOT.exists():
        pythonpath_entries.append(str(LIBERO_ROOT))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    cache_root = env.get("QUANTVLA_CACHE_ROOT", "/work/mingze/.cache/quantvla")
    env["QUANTVLA_CACHE_ROOT"] = cache_root
    env["HF_HOME"] = f"{cache_root}/huggingface"
    env["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = f"{cache_root}/huggingface/transformers"
    env["TORCH_HOME"] = f"{cache_root}/torch"
    env["XDG_CACHE_HOME"] = f"{cache_root}/xdg"
    env["LIBERO_CONFIG_PATH"] = env.get("LIBERO_CONFIG_PATH", "/work/mingze/.libero")
    env["LIBERO_ROOT"] = str(LIBERO_ROOT)
    env["MPLCONFIGDIR"] = f"{cache_root}/matplotlib"

    for key in (
        env["HF_HOME"],
        env["HUGGINGFACE_HUB_CACHE"],
        env["TRANSFORMERS_CACHE"],
        env["TORCH_HOME"],
        env["XDG_CACHE_HOME"],
        env["LIBERO_CONFIG_PATH"],
        env["MPLCONFIGDIR"],
    ):
        Path(key).mkdir(parents=True, exist_ok=True)
    return env


def configure_duquant_env(env: dict[str, str], wbits: int, abits: int, packdir: str) -> None:
    env["GR00T_DUQUANT_DEBUG"] = env.get("GR00T_DUQUANT_DEBUG", "1")
    env["GR00T_DUQUANT_SCOPE"] = env.get("GR00T_DUQUANT_SCOPE", "")
    env["GR00T_DUQUANT_INCLUDE"] = env.get("GR00T_DUQUANT_INCLUDE", (
        r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r"|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
    ))
    env["GR00T_DUQUANT_EXCLUDE"] = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
    )
    env["GR00T_DUQUANT_WBITS_DEFAULT"] = str(wbits)
    env["GR00T_DUQUANT_ABITS"] = str(abits)
    env["GR00T_DUQUANT_BLOCK"] = env.get("GR00T_DUQUANT_BLOCK", "64")
    env["GR00T_DUQUANT_BLOCK_OUT"] = env.get("GR00T_DUQUANT_BLOCK_OUT", env["GR00T_DUQUANT_BLOCK"])
    env["GR00T_DUQUANT_PERMUTE"] = env.get("GR00T_DUQUANT_PERMUTE", "0")
    env["GR00T_DUQUANT_ROW_ROT"] = env.get("GR00T_DUQUANT_ROW_ROT", "restore")
    env["GR00T_DUQUANT_ACT_PCT"] = env.get("GR00T_DUQUANT_ACT_PCT", "99.9")
    env["GR00T_DUQUANT_CALIB_STEPS"] = env.get("GR00T_DUQUANT_CALIB_STEPS", "32")
    env["GR00T_DUQUANT_LS"] = env.get("GR00T_DUQUANT_LS", "0.15")
    env["GR00T_DUQUANT_PACKDIR"] = packdir
    env["GR00T_ATM_ENABLE"] = "0"
    env["GR00T_OHB_ENABLE"] = "0"
    env.pop("GR00T_ATM_ALPHA_PATH", None)
    env.pop("GR00T_ATM_SCOPE", None)
    env.pop("GR00T_OHB_FALLBACK", None)
    env.pop("GR00T_OHB_SCOPE", None)
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_CUDA_GRAPH_DISABLE"] = "1"
    env["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"


def run_shard(
    shard_idx: int,
    gpu: str,
    port: int,
    task_ids: list[int],
    task_suite: str,
    model_path: str,
    data_config: str,
    num_trials: int,
    num_steps_wait: int,
    denoising_steps: int,
    wbits: int,
    abits: int,
    output_root: Path,
    packdir: str,
    headless: bool,
) -> None:
    base_env = make_base_env()
    base_env["CUDA_VISIBLE_DEVICES"] = gpu
    configure_duquant_env(base_env, wbits=wbits, abits=abits, packdir=packdir)

    logs_dir = output_root / "logs"
    eval_log_dir = logs_dir / "eval"
    summaries_dir = output_root / "summaries"
    rollout_dir = output_root / "rollouts" / f"shard_{shard_idx}"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    server_log_path = logs_dir / f"server_shard_{shard_idx}.log"
    eval_stdout_log_path = logs_dir / f"eval_shard_{shard_idx}.stdout.log"
    summary_json_path = summaries_dir / f"shard_{shard_idx}.json"

    with open(server_log_path, "w") as server_log:
        server_proc = subprocess.Popen(
            [
                GROOT_PY,
                str(QUANTVLA_ROOT / "scripts" / "inference_service.py"),
                "--model_path",
                model_path,
                "--server",
                "--data_config",
                data_config,
                "--denoising-steps",
                str(denoising_steps),
                "--port",
                str(port),
                "--embodiment-tag",
                "new_embodiment",
            ],
            cwd=str(QUANTVLA_ROOT),
            env=base_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    try:
        wait_for_port(port)
        eval_cmd = [
            LIBERO_PY,
            "run_libero_eval.py",
            "--task_suite_name",
            task_suite,
            "--num_trials_per_task",
            str(num_trials),
            "--num_steps_wait",
            str(num_steps_wait),
            "--port",
            str(port),
            "--task_ids",
            *[str(task_id) for task_id in task_ids],
            "--log_dir",
            str(eval_log_dir),
            "--log_suffix",
            f"shard_{shard_idx}",
            "--rollout_dir",
            str(rollout_dir),
            "--summary_json",
            str(summary_json_path),
        ]
        if headless:
            eval_cmd.append("--headless")
        eval_cmd.append("--no-save-videos")

        with open(eval_stdout_log_path, "w") as eval_log:
            subprocess.run(
                eval_cmd,
                cwd=str(QUANTVLA_ROOT / "examples" / "Libero" / "eval"),
                env=base_env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=30)


def merge_summaries(
    output_root: Path,
    task_suite: str,
    model_path: str,
    data_config: str,
    gpu_list: list[str],
    wbits: int,
    abits: int,
    num_trials: int,
    num_steps_wait: int,
) -> None:
    summary_files = sorted((output_root / "summaries").glob("shard_*.json"))
    if not summary_files:
        raise RuntimeError("No shard summary JSON files found")

    summaries = [json.loads(path.read_text()) for path in summary_files]
    total_episodes = sum(item["total_episodes"] for item in summaries)
    total_successes = sum(item["total_successes"] for item in summaries)
    task_rows = []
    for item in summaries:
        task_rows.extend(item.get("task_summaries", []))
    task_rows.sort(key=lambda row: row["task_id"])

    merged = {
        "task_suite_name": task_suite,
        "model_path": model_path,
        "data_config": data_config,
        "gpu_list": gpu_list,
        "wbits": wbits,
        "abits": abits,
        "num_trials_per_task": num_trials,
        "num_steps_wait": num_steps_wait,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": (float(total_successes) / float(total_episodes)) if total_episodes else 0.0,
        "num_task_shards": len(summaries),
        "task_summaries": task_rows,
        "shard_summaries": summaries,
    }
    (output_root / "merged_summary.json").write_text(json.dumps(merged, indent=2))

    lines = [
        "# LIBERO DuQuant Benchmark Summary",
        "",
        f"- Task suite: {task_suite}",
        f"- GPUs: {','.join(gpu_list)}",
        f"- Quantization: W{wbits}A{abits} pure DuQuant baseline",
        f"- Episodes: {total_successes}/{total_episodes} successes ({merged['total_success_rate'] * 100:.1f}%)",
        "",
        "| task_id | success_rate | successes | episodes | task_description |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task_id']} | {row['success_rate'] * 100:.1f}% | {row['successes']} | {row['episodes']} | {row['task_description']} |"
        )
    (output_root / "merged_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    task_suite = env_str("TASK_SUITE", "libero_10")
    gpu_list = [token.strip() for token in env_str("GPU_LIST", "0,1,2,3,4,5,6,7").split(",") if token.strip()]
    if not gpu_list:
        raise SystemExit("GPU_LIST is empty")

    default_model_path, data_config, num_tasks = suite_defaults(task_suite)
    model_path = env_str("MODEL_PATH", default_model_path)
    num_trials = env_int("NUM_TRIALS_PER_TASK", 5)
    num_steps_wait = env_int("NUM_STEPS_WAIT", 10)
    denoising_steps = env_int("DENOISING_STEPS", 8)
    wbits = env_int("WBITS", 3)
    abits = env_int("ABITS", 8)
    headless = env_str("HEADLESS", "1") == "1"
    port_base = env_int("PORT_BASE", 5600)
    output_root = Path(
        env_str(
            "OUTPUT_ROOT",
            str(
                QUANTVLA_ROOT
                / "results"
                / f"libero_duquant_{task_suite}_w{wbits}a{abits}_gpu{'_'.join(gpu_list)}"
            ),
        )
    )
    packdir = env_str(
        "PACKDIR", str(QUANTVLA_ROOT / f"duquant_packed_paperalign_{task_suite}_w{wbits}a{abits}")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)
    (output_root / "rollouts").mkdir(parents=True, exist_ok=True)

    task_shards = shard_tasks(num_tasks=num_tasks, num_shards=len(gpu_list))
    (output_root / "task_shards.json").write_text(json.dumps(task_shards, indent=2))

    print("========================================")
    print("LIBERO DuQuant Benchmark Launcher")
    print("========================================")
    print(f"Task suite : {task_suite}")
    print(f"Model      : {model_path}")
    print(f"Data config: {data_config}")
    print(f"Output     : {output_root}")
    print(f"GPUs       : {','.join(gpu_list)}")
    print(f"Trials     : {num_trials}")
    print(f"Steps wait : {num_steps_wait}")
    print(f"WBITS/ABITS: {wbits}/{abits}")
    print(f"Tasks      : {num_tasks}")
    print(f"Packdir    : {packdir}")
    print("========================================")

    shard_procs: list[subprocess.Popen] = []
    shard_driver_logs: list[Path] = []

    for shard_idx, gpu in enumerate(gpu_list):
        task_ids = task_shards[shard_idx]
        if not task_ids:
            print(f"[Skip] shard={shard_idx} gpu={gpu} has no tasks")
            continue

        port = port_base + shard_idx
        driver_log_path = output_root / "logs" / f"driver_shard_{shard_idx}.log"
        shard_driver_logs.append(driver_log_path)
        cmd = [
            GROOT_PY,
            str(Path(__file__).resolve()),
            "--run-shard",
            str(shard_idx),
            gpu,
            str(port),
            json.dumps(task_ids),
            task_suite,
            model_path,
            data_config,
            str(num_trials),
            str(num_steps_wait),
            str(denoising_steps),
            str(wbits),
            str(abits),
            str(output_root),
            packdir,
            "1" if headless else "0",
        ]
        with open(driver_log_path, "w") as driver_log:
            proc = subprocess.Popen(cmd, cwd=str(QUANTVLA_ROOT), stdout=driver_log, stderr=subprocess.STDOUT)
        shard_procs.append(proc)
        print(f"[Launch] shard={shard_idx} gpu={gpu} port={port} tasks={task_ids}")

    try:
        for proc in shard_procs:
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"Shard process exited with code {return_code}")
    except KeyboardInterrupt:
        for proc in shard_procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        raise

    merge_summaries(
        output_root=output_root,
        task_suite=task_suite,
        model_path=model_path,
        data_config=data_config,
        gpu_list=gpu_list,
        wbits=wbits,
        abits=abits,
        num_trials=num_trials,
        num_steps_wait=num_steps_wait,
    )

    print("========================================")
    print("Finished")
    print(f"Output     : {output_root}")
    print(f"Merged JSON: {output_root / 'merged_summary.json'}")
    print(f"Merged MD  : {output_root / 'merged_summary.md'}")
    print(f"Logs       : {output_root / 'logs'}")
    print("========================================")
    return 0


def shard_entry(argv: list[str]) -> int:
    if len(argv) != 16:
        raise SystemExit(f"Expected 16 shard arguments, got {len(argv)}")
    (
        shard_idx,
        gpu,
        port,
        task_ids_json,
        task_suite,
        model_path,
        data_config,
        num_trials,
        num_steps_wait,
        denoising_steps,
        wbits,
        abits,
        output_root,
        packdir,
        headless,
    ) = argv[1:]
    run_shard(
        shard_idx=int(shard_idx),
        gpu=gpu,
        port=int(port),
        task_ids=json.loads(task_ids_json),
        task_suite=task_suite,
        model_path=model_path,
        data_config=data_config,
        num_trials=int(num_trials),
        num_steps_wait=int(num_steps_wait),
        denoising_steps=int(denoising_steps),
        wbits=int(wbits),
        abits=int(abits),
        output_root=Path(output_root),
        packdir=packdir,
        headless=headless == "1",
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-shard":
        raise SystemExit(shard_entry(sys.argv[1:]))
    raise SystemExit(main())
