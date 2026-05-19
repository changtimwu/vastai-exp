#!/usr/bin/env python3
"""
Hands-off llama.cpp benchmarking on rented vast.ai GPUs.

Example:
    ./bench.py https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF \\
        --gpu rtx-5060ti --num-gpus 2 \\
        --bin llama-cli \\
        --params "--spec-draft-p-min 0.75 --spec-type draft-mtp"

Prerequisites:
    - vastai API key set:  vastai set api-key <KEY>
    - SSH key registered:  vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VASTAI = ROOT / "venv" / "bin" / "vastai"
REMOTE_SETUP = ROOT / "remote_setup.sh"

# user-friendly aliases -> vast.ai's gpu_name field (spaces become _)
GPU_ALIASES = {
    "rtx-5060ti": "RTX_5060_Ti",
    "rtx-5070ti": "RTX_5070_Ti",
    "rtx-5080": "RTX_5080",
    "rtx-5090": "RTX_5090",
    "rtx-4090": "RTX_4090",
    "rtx-4080": "RTX_4080",
    "rtx-3090": "RTX_3090",
    "h100": "H100_SXM",
    "h100-pcie": "H100_PCIE",
    "a100": "A100_SXM4",
}


def vastai(*args: str) -> str:
    out = subprocess.run([str(VASTAI), *args], check=True, capture_output=True, text=True)
    return out.stdout


def normalize_gpu(name: str) -> str:
    key = name.lower().replace("_", "-").replace(" ", "-")
    return GPU_ALIASES.get(key, name.replace("-", "_").replace(" ", "_"))


def search_offers(gpu: str, num_gpus: int, min_disk_gb: int, max_hourly: float | None) -> list[dict]:
    parts = [
        f"gpu_name={gpu}",
        f"num_gpus={num_gpus}",
        "verified=true",
        "rentable=true",
        f"disk_space>={min_disk_gb}",
        "reliability>0.98",
        "cuda_max_good>=12.8",
        "inet_down>=200",
    ]
    if max_hourly is not None:
        parts.append(f"dph_total<={max_hourly}")
    raw = vastai("search", "offers", " ".join(parts), "-o", "dlperf_per_dphtotal-", "--raw")
    return json.loads(raw)


def create_instance(offer_id: int, image: str, disk_gb: int, onstart_cmd: str) -> int:
    raw = vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--ssh",
        "--direct",
        "--onstart-cmd", onstart_cmd,
        "--cancel-unavail",
        "--raw",
    )
    resp = json.loads(raw)
    if not resp.get("success", True):
        raise RuntimeError(f"create failed: {resp}")
    return resp["new_contract"]


def show_instance(instance_id: int) -> dict:
    raw = vastai("show", "instance", str(instance_id), "--raw")
    return json.loads(raw)


def wait_ssh(instance_id: int, timeout: int = 900) -> dict:
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        info = show_instance(instance_id)
        status = info.get("actual_status")
        if status != last_status:
            print(f"  status: {status}")
            last_status = status
        ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        ssh_port = info.get("ssh_port") or _direct_ssh_port(info)
        if status == "running" and ssh_host and ssh_port and _ssh_ping(ssh_host, ssh_port):
            info["_ssh_host"] = ssh_host
            info["_ssh_port"] = int(ssh_port)
            return info
        time.sleep(10)
    raise TimeoutError(f"instance {instance_id} not SSH-ready within {timeout}s")


def _direct_ssh_port(info: dict) -> int | None:
    # When --direct is used, port 22 in container is mapped on the host.
    ports = info.get("ports") or {}
    mapping = ports.get("22/tcp")
    if mapping and isinstance(mapping, list) and mapping:
        return int(mapping[0].get("HostPort", 0)) or None
    return None


def _ssh_ping(host: str, port: int) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5", "-o", "UserKnownHostsFile=/dev/null",
             "-p", str(port), f"root@{host}", "true"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def destroy_instance(instance_id: int) -> None:
    try:
        vastai("destroy", "instance", str(instance_id))
        print(f"  destroyed {instance_id}")
    except subprocess.CalledProcessError as e:
        print(f"WARN: destroy failed for {instance_id}: {e.stderr or e}", file=sys.stderr)


def ssh_run(info: dict, remote_cmd: str) -> int:
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(info["_ssh_port"]), f"root@{info['_ssh_host']}",
        remote_cmd,
    ]
    return subprocess.run(cmd).returncode


def scp(info: dict, src: str, dst: str, direction: str = "to") -> None:
    host = info["_ssh_host"]
    port = info["_ssh_port"]
    base = ["scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-P", str(port)]
    if direction == "to":
        subprocess.run(base + [src, f"root@{host}:{dst}"], check=True)
    else:
        subprocess.run(base + [f"root@{host}:{src}", dst], check=True)


def parse_hf_url(url: str) -> str:
    m = re.match(r"https?://huggingface\.co/([^/?#]+/[^/?#]+)", url)
    if not m:
        raise ValueError(f"not a Hugging Face model URL: {url}")
    return m.group(1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_url", help="Hugging Face model URL, e.g. https://huggingface.co/owner/repo")
    p.add_argument("--gpu", required=True, help="GPU type, e.g. rtx-5060ti")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--params", default="", help="extra flags passed verbatim to the benchmark binary")
    p.add_argument("--bin", default="llama-bench",
                   help="llama.cpp binary to run (llama-bench, llama-cli, llama-server)")
    p.add_argument("--quant", default="Q4_K_M", help="quant substring to match in GGUF filenames")
    p.add_argument("--disk", type=int, default=120, help="instance disk size in GB")
    p.add_argument("--image", default="nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04")
    p.add_argument("--max-hourly", type=float, help="reject offers above this $/hr")
    p.add_argument("--yes", action="store_true", help="skip interactive offer confirmation")
    p.add_argument("--keep", action="store_true", help="leave the instance running on exit")
    args = p.parse_args()

    gpu = normalize_gpu(args.gpu)
    repo_id = parse_hf_url(args.model_url)

    print(f"GPU       : {gpu} x{args.num_gpus}")
    print(f"Model     : {repo_id} (quant match: {args.quant})")
    print(f"Binary    : {args.bin}")
    print(f"Params    : {args.params}")

    print("\nSearching offers...")
    offers = search_offers(gpu, args.num_gpus, args.disk, args.max_hourly)
    if not offers:
        print(f"no matching offers for {gpu} x{args.num_gpus}", file=sys.stderr)
        return 1

    pick = offers[0]
    print(f"\nTop offer: id={pick['id']}  ${pick['dph_total']:.3f}/hr  "
          f"DLP={pick['dlperf']:.1f}  DLP/$={pick['dlperf_per_dphtotal']:.1f}  "
          f"{pick['cpu_name'].split()[0]} {pick['cpu_cores']}c  "
          f"{pick['geolocation']}")
    if not args.yes and input("Rent this? [y/N] ").strip().lower() != "y":
        return 2

    # Stage benchmark config files via onstart so they exist before we SSH in.
    onstart = "; ".join([
        f"echo {shlex.quote(repo_id)} > /root/.bench_repo",
        f"echo {shlex.quote(args.quant)} > /root/.bench_quant",
        f"echo {shlex.quote(args.bin)}   > /root/.bench_bin",
        f"echo {shlex.quote(args.params)} > /root/.bench_params",
    ])

    print("\nCreating instance...")
    instance_id = create_instance(pick["id"], args.image, args.disk, onstart)
    print(f"  instance id: {instance_id}")

    try:
        print("Waiting for SSH...")
        info = wait_ssh(instance_id)
        print(f"  ssh root@{info['_ssh_host']} -p {info['_ssh_port']}")

        print("Uploading remote_setup.sh...")
        scp(info, str(REMOTE_SETUP), "/root/remote_setup.sh", direction="to")

        print("Running benchmark on remote (streamed below)...")
        rc = ssh_run(info, "bash /root/remote_setup.sh 2>&1 | tee /root/bench.log")
        if rc != 0:
            print(f"remote_setup.sh exited with {rc}", file=sys.stderr)

        print("\nFetching log + summary...")
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        scp(info, "/root/bench.log", str(out_dir / f"{instance_id}.log"), direction="from")
        scp(info, "/root/bench.out", str(out_dir / f"{instance_id}.out"), direction="from")
        print(f"\n=== {out_dir / f'{instance_id}.out'} ===")
        print((out_dir / f"{instance_id}.out").read_text())
        return rc
    finally:
        if args.keep:
            print(f"\nInstance {instance_id} left running (--keep).")
        else:
            print(f"\nDestroying instance {instance_id}...")
            destroy_instance(instance_id)


if __name__ == "__main__":
    sys.exit(main())
