"""
inference.py — MetaMind OpenEnv Baseline Inference Script
==========================================================
Required by OpenEnv spec. Runs the full CB → MAB → QL pipeline
against a set of test tasks and reports reproducible scores.

Usage:
    python inference.py
    python inference.py --task "Explain gradient descent"
    python inference.py --url http://localhost:8000
"""

import argparse
import json
import sys
import time
import requests

DEFAULT_URL = "http://localhost:8000"

TEST_TASKS = [
    "What is machine learning?",
    "Explain gradient descent in simple terms.",
    "How does a neural network learn?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of overfitting.",
]


def run_episode(base_url: str, task: str, verbose: bool = True) -> dict:
    """Run one full episode: reset → step × 3 (or until done)."""

    # Reset
    r = requests.post(f"{base_url}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    reset_data = r.json()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"Run ID: {reset_data.get('info', {}).get('run_id', 'N/A')}")
        print(f"{'='*60}")

    steps = []
    total_reward = 0.0
    done = False

    for i in range(3):
        if done:
            break

        step_r = requests.post(
            f"{base_url}/step",
            json={"action": {"type": "reason", "content": task}},
            timeout=120,
        )
        step_r.raise_for_status()
        step_data = step_r.json()

        info   = step_data.get("info", {})
        reward = step_data.get("reward", 0.0)
        done   = step_data.get("done", False)
        total_reward += reward

        steps.append({
            "step":      i + 1,
            "algorithm": info.get("algorithm", ""),
            "score":     info.get("score", 0),
            "reward":    reward,
            "passed":    info.get("passed", False),
        })

        if verbose:
            status = "✓ PASS" if info.get("passed") else "✗ FAIL"
            print(f"\n[Step {i+1}] {info.get('algorithm','')} ({info.get('level','')})")
            print(f"  Status : {status}")
            print(f"  Score  : {info.get('score', 0)*100:.1f}%  "
                  f"(threshold {info.get('threshold', 0)*100:.0f}%)")
            print(f"  Reward : {reward:+.3f}")
            print(f"  Reason : {info.get('reason','')}")
            if info.get("response"):
                print(f"  Answer : {info['response'][:200]}{'...' if len(info.get('response',''))>200 else ''}")

    first_score = steps[0]["score"] if steps else 0
    last_score  = steps[-1]["score"] if steps else 0
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
        print(f"\n{'─'*60}")
        print(f"  Winner     : {result['winning_algorithm']}")
        print(f"  Final Score: {last_score*100:.1f}%")
        print(f"  Net Reward : {total_reward:+.3f}")
        print(f"  Adapt Gain : {adapt_gain*100:.1f}%")
        print(f"  Attempts   : {len(steps)}")
        print(f"  Success    : {result['success']}")

    return result


def run_baseline(base_url: str, tasks: list[str]) -> dict:
    """Run all tasks and compute aggregate metrics."""
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
            time.sleep(1)  # avoid rate limiting
        except Exception as e:
            print(f"\n[ERROR] Task failed: {e}")
            results.append({"task": task, "error": str(e)})

    n = len(tasks)
    summary = {
        "total_tasks":       n,
        "success_rate":      round(total_success / n, 4),
        "avg_reward":        round(total_reward / n, 4),
        "avg_adapt_gain":    round(total_adapt / n, 4),
        "results":           results,
    }

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks        : {n}")
    print(f"  Success Rate : {summary['success_rate']*100:.1f}%")
    print(f"  Avg Reward   : {summary['avg_reward']:+.3f}")
    print(f"  Avg Adapt    : {summary['avg_adapt_gain']*100:.1f}%")
    print(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(description="MetaMind Baseline Inference")
    parser.add_argument("--url",  default=DEFAULT_URL, help="Backend base URL")
    parser.add_argument("--task", default=None,        help="Single task to run")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TEST_TASKS

    try:
        # Health check
        r = requests.get(args.url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach backend at {args.url}: {e}")
        print("Start the server: uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    run_baseline(args.url, tasks)


if __name__ == "__main__":
    main()
