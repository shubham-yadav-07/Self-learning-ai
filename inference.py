"""
inference.py — MetaMind OpenEnv Baseline Inference Script
==========================================================
Required file at repo root by OpenEnv spec.
Runs the full CB → MAB → QL pipeline against test tasks.

Usage:
    python inference.py
    python inference.py --task "Explain neural networks"
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
    """Run one full episode: POST /reset → POST /step (up to 3 times)."""

    # ── Reset ──────────────────────────────────────────────────
    try:
        r = requests.post(
            f"{base_url}/reset",
            json={"task": task},
            timeout=30,
        )
        r.raise_for_status()
        reset_data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] /reset failed for task '{task}': {e}")
        return {
            "task": task,
            "steps": [],
            "total_reward": 0.0,
            "winning_algorithm": None,
            "final_score": 0.0,
            "adaptation_gain": 0.0,
            "attempts_used": 0,
            "success": False,
            "error": str(e),
        }
    except (ValueError, KeyError) as e:
        print(f"[ERROR] /reset response parse error for task '{task}': {e}")
        return {
            "task": task,
            "steps": [],
            "total_reward": 0.0,
            "winning_algorithm": None,
            "final_score": 0.0,
            "adaptation_gain": 0.0,
            "attempts_used": 0,
            "success": False,
            "error": str(e),
        }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task   : {task}")
        print(f"Run ID : {reset_data.get('info', {}).get('run_id', 'N/A')}")
        print(f"{'='*60}")

    steps = []
    total_reward = 0.0
    done = reset_data.get("done", False)

    # ── Steps ──────────────────────────────────────────────────
    for i in range(3):
        if done:
            break

        try:
            step_r = requests.post(
                f"{base_url}/step",
                json={"action": {"type": "reason", "content": task}},
                timeout=120,
            )
            step_r.raise_for_status()
            step_data = step_r.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] /step {i+1} failed: {e}")
            break
        except (ValueError, KeyError) as e:
            print(f"[ERROR] /step {i+1} response parse error: {e}")
            break

        try:
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
                print(f"\n[Step {i+1}] {algorithm} ({level})")
                print(f"  Status    : {status}")
                print(f"  Score     : {score*100:.1f}%  (threshold {threshold*100:.0f}%)")
                print(f"  Reward    : {reward:+.3f}")
                print(f"  Reason    : {reason}")
                if response:
                    snippet = response[:200]
                    print(f"  Response  : {snippet}{'...' if len(response) > 200 else ''}")

        except Exception as e:
            print(f"[ERROR] Step {i+1} processing error: {e}")
            break

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
        print(f"\n{'─'*60}")
        print(f"  Winner      : {result['winning_algorithm']}")
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
            if not result.get("error"):
                total_success += int(result["success"])
                total_reward  += result["total_reward"]
                total_adapt   += result["adaptation_gain"]
            time.sleep(1)
        except Exception as e:
            print(f"\n[ERROR] Task failed unexpectedly: {e}")
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

    # Health check — non-fatal: warn and continue rather than exiting
    try:
        r = requests.get(args.url, timeout=10)
        if r.status_code not in (200, 404):
            # 404 on root is fine — server is reachable
            print(f"[WARN] Backend returned HTTP {r.status_code} — proceeding anyway.")
    except requests.exceptions.ConnectionError:
        print(f"[WARN] Cannot reach {args.url} — server may not be running.")
        print("Start server: uvicorn main:app --host 0.0.0.0 --port 8000")
        # Do NOT sys.exit here — let run_baseline handle per-task errors gracefully
    except Exception as e:
        print(f"[WARN] Health check error: {e} — proceeding anyway.")

    try:
        run_baseline(args.url, tasks)
    except Exception as e:
        print(f"[ERROR] run_baseline crashed unexpectedly: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
