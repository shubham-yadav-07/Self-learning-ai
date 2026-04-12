"""
inference.py — MetaMind OpenEnv Baseline Inference Script
==========================================================
Required file at repo root by OpenEnv spec.
Runs the full CB → MAB → QL pipeline against test tasks.

Usage:
    python inference.py
    python inference.py --task "Explain neural networks"
    python inference.py --url http://localhost:7860
"""

import argparse
import json
import sys
import time
import requests
import os
from openai import OpenAI

DEFAULT_URL  = "http://localhost:7860"
def get_client():
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    if not base_url or not api_key:
        print("[WARNING] No proxy, using fallback", flush=True)
        return None

    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(prompt):
    client = get_client()

    if client is None:
        return prompt  # fallback

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", flush=True)
        return prompt
TEST_TASKS   = [
    "What is machine learning?",
    "Explain gradient descent in simple terms.",
    "How does a neural network learn?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of overfitting.",
]


def run_episode(base_url: str, task: str, verbose: bool = True) -> dict:
    """Run one full episode: POST /reset → POST /step (up to 3 times)."""

    # ── Reset ──────────────────────────────────────────────────
    r = requests.post(
        f"{base_url}/reset",
        json={"task": task},
        timeout=30,
    )
    r.raise_for_status()
    reset_data = r.json()

    if verbose:
        print(f"[START] task={task}", flush=True)
        print(f"Run ID : {reset_data.get('info', {}).get('run_id', 'N/A')}")
        print(f"{'='*60}")

    steps        = []
    total_reward = 0.0
    done         = reset_data.get("done", False)

    # ── Steps ──────────────────────────────────────────────────
    for i in range(3):
        if done:
            break
        llm_output = call_llm(task)

        step_r = requests.post(
            f"{base_url}/step",
            json={"action": {"type": "reason", "content": llm_output}},
            timeout=120,
        )
        step_r.raise_for_status()
        step_data = step_r.json()

        reward = step_data.get("reward", 0.0)
        done   = step_data.get("done",   False)
        info   = step_data.get("info",   {})
        obs    = step_data.get("observation", {})

        # info fields (fallback to obs fields)
        algorithm = info.get("algorithm") or obs.get("algorithm", f"Level-{i+1}")
        level     = info.get("level")     or obs.get("level",     "")
        score     = info.get("score",     obs.get("reward") or 0.0)
        passed    = info.get("passed",    done)
        threshold = info.get("threshold", 0.0)
        reason    = info.get("reason",    "")
        response  = info.get("response",  "")

        total_reward += reward
        steps.append({
            "step":      i + 1,
            "algorithm": algorithm,
            "level":     level,
            "score":     score,
            "reward":    reward,
            "passed":    passed,
        })

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"[STEP] step={i+1} reward={reward}", flush=True)
            print(f"  Reason    : {reason}")
            if response:
                snippet = response[:200]
                print(f"  Response  : {snippet}{'...' if len(response) > 200 else ''}")

    first_score = steps[0]["score"] if steps else 0.0
    last_score  = steps[-1]["score"] if steps else 0.0
    adapt_gain  = max(0.0, last_score - first_score)

    result = {
        "task":              task,
        "steps":             steps,
        "total_reward":      round(total_reward, 4),
        "winning_algorithm": steps[-1]["algorithm"] if steps else None,
        "final_score":       last_score,
        "adaptation_gain":   round(adapt_gain, 4),
        "attempts_used":     len(steps),
        "success":           steps[-1]["passed"] if steps else False,
    }

    if verbose:
        print(f"[END] task={task} score={last_score:.2f} steps={len(steps)}", flush=True)
        print(f"  Final Score : {last_score*100:.1f}%")
        print(f"  Net Reward  : {total_reward:+.3f}")
        print(f"  Adapt Gain  : {adapt_gain*100:.1f}%")
        print(f"  Success     : {result['success']}")

    return result


def run_baseline(base_url: str, tasks: list) -> dict:
    """Run all tasks and print aggregate metrics."""
    print(f"\nMetaMind Baseline Inference")
    print(f"Backend : {base_url}")
    print(f"Tasks   : {len(tasks)}")

    results       = []
    total_success = 0
    total_reward  = 0.0
    total_adapt   = 0.0

    for task in tasks:
        try:
            result = run_episode(base_url, task, verbose=True)
            results.append(result)
            total_success += int(result["success"])
            total_reward  += result["total_reward"]
            total_adapt   += result["adaptation_gain"]
            time.sleep(1)
        except Exception as e:
            print(f"\n[ERROR] Task failed: {e}")
            results.append({"task": task, "error": str(e)})

    n = max(len(tasks), 1)
    summary = {
        "total_tasks":    n,
        "success_rate":   round(total_success / n, 4),
        "avg_reward":     round(total_reward / n, 4),
        "avg_adapt_gain": round(total_adapt / n, 4),
        "results":        results,
    }

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks        : {n}")
    print(f"  Success Rate : {summary['success_rate']*100:.1f}%")
    print(f"  Avg Reward   : {summary['avg_reward']:+.3f}")
    print(f"  Avg Adapt    : {summary['avg_adapt_gain']*100:.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="MetaMind Baseline Inference")
    parser.add_argument("--url",  default=DEFAULT_URL, help="Backend base URL")
    parser.add_argument("--task", default=None,        help="Single task to run")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TEST_TASKS

    # Health check
    try:
        r = requests.get(args.url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach {args.url}: {e}")
        
        print("Start server: uvicorn main:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    run_baseline(args.url, tasks)


if __name__ == "__main__":
    main()
