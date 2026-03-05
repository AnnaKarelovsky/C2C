"""Quick test: launch minisglang, run 5 Countdown eval samples, show responses.

Usage:
    CUDA_VISIBLE_DEVICES=1 python script/optimize/example/test_countdown.py \
        --model Qwen/Qwen3-1.7B

    # With a checkpoint:
    CUDA_VISIBLE_DEVICES=1 python script/optimize/example/test_countdown.py \
        --model Qwen/Qwen3-1.7B --opt-cache path/to/opt_cache
"""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import time

import requests

PORT = 30010


def wait_for_server(port: int, timeout: int = 180):
    """Block until the server at *port* responds to health checks."""
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server on port {port} not ready after {timeout}s")


def launch_server(model: str, opt_cache: str | None = None):
    """Launch minisglang and return the subprocess."""
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model-path", model,
        "--port", str(PORT),
        "--memory-ratio", "0.8",
        "--tool-call-parser", "qwen",
    ]
    if opt_cache:
        cmd += ["--opt-cache", opt_cache]

    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Ensure cleanup on exit
    def _kill():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)

    atexit.register(_kill)

    print("Waiting for server to be ready...")
    wait_for_server(PORT)
    print("Server ready.\n")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--opt-cache", default=None, help="Opt cache dir for minisglang")
    parser.add_argument("--n-problems", type=int, default=5)
    args = parser.parse_args()

    proc = launch_server(args.model, args.opt_cache)

    from rosetta.optimize.interface.countdown import (
        CountdownInterface,
        _countdown_reward,
        _format_completion,
    )
    from rosetta.optimize.train_utils import RolloutEngine

    engine = RolloutEngine(f"http://localhost:{PORT}", args.model)
    interface = CountdownInterface(engine=engine)
    eval_data = interface._get_eval_data()[:args.n_problems]
    interface._eval_data = eval_data

    # Run eval
    interface.eval_fn(global_step=0, n_samples=1)

    # Print formatted responses
    from rosetta.optimize.interface.countdown import PROMPT_TOOL

    prompts = [prompt for _, _, prompt in eval_data]
    tools_list = [PROMPT_TOOL] * len(prompts)
    extra = {"chat_template_kwargs": interface.tmpl_kwargs} if interface.tmpl_kwargs else {}
    completions = engine.generate(
        prompts, max_tokens=2048, temperature=0.9,
        tools_list=tools_list, **extra,
    )

    for i, (target, nums, prompt) in enumerate(eval_data):
        c = completions[i]
        score = _countdown_reward(c, target, nums)
        formatted = _format_completion(c, prompt, interface.tokenizer, interface.tmpl_kwargs)
        print("=" * 70)
        print(f"Problem {i+1}: {nums} -> {target}  (reward={score})")
        print("-" * 70)
        print(formatted)
        print()

    # Clean up
    proc.terminate()
    proc.wait()


if __name__ == "__main__":
    main()
