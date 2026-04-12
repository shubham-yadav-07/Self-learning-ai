"""
MetaMind — OpenEnv Self-Improvement Environment
Uses OpenRouter API (sk-or-v1-... key)
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os, json, math, random, time, uuid, re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests as req_lib
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct:free"  # free tier on OpenRouter

# ─── App ────────────────────────────────────────────────────────
app = FastAPI(title="MetaMind", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Algorithms ─────────────────────────────────────────────────
ALGO_ORDER = ["contextual_bandit", "multi_armed_bandit", "q_learning"]

ALGORITHMS = {
    "contextual_bandit": {
        "name": "Contextual Bandit", "level": "easy",
        "threshold": 0.60, "r_pass": 1.0, "r_fail": -0.5, "pen": 0.05,
        "system": (
            "You are a Level-1 Contextual Bandit AI. "
            "Give the most direct, concise answer in 2-3 sentences. No elaboration."
        ),
    },
    "multi_armed_bandit": {
        "name": "Multi-Armed Bandit", "level": "medium",
        "threshold": 0.75, "r_pass": 1.5, "r_fail": -0.3, "pen": 0.03,
        "system": (
            "You are a Level-2 Multi-Armed Bandit AI using UCB1 exploration. "
            "The previous attempt was insufficient. Choose the best strategy arm "
            "(direct / step-by-step / analogy / structured) and answer in 4-6 sentences."
        ),
    },
    "q_learning": {
        "name": "Q-Learning", "level": "hard",
        "threshold": 0.85, "r_pass": 2.0, "r_fail": -0.1, "pen": 0.01,
        "system": (
            "You are a Level-3 Q-Learning AI optimising cumulative reward. "
            "Previous attempts failed. Produce your absolute best, comprehensive, "
            "well-structured answer. Quality is the only objective."
        ),
    },
}

# ─── Agent state (persists per process) ─────────────────────────
class MAB:
    ARMS = ["direct", "step_by_step", "analogy", "structured", "first_principles"]
    C = math.sqrt(2)
    def __init__(self):
        self.counts = {a: 0   for a in self.ARMS}
        self.values = {a: 0.0 for a in self.ARMS}
        self.total  = 0
    def select(self):
        self.total += 1
        for a in self.ARMS:
            if self.counts[a] == 0: return a
        return max(self.ARMS, key=lambda a:
            self.values[a] + self.C * math.sqrt(math.log(self.total)/self.counts[a]))
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class QTable:
    ACTIONS = ["comprehensive", "structured", "examples", "concise", "socratic"]
    def __init__(self): self.table: Dict = {}; self.alpha=0.15; self.gamma=0.9; self.eps=0.1
    def _k(self, i, s): return f"a{i}_s{'none' if s is None else int(s*10)}"
    def select(self, i, s):
        if random.random() < self.eps: return random.choice(self.ACTIONS)
        k = self._k(i, s)
        return max(self.ACTIONS, key=lambda a: self.table.get(k,{}).get(a,0.0))
    def update(self, i, s, action, reward, ns):
        k, nk = self._k(i,s), self._k(i+1,ns)
        if k not in self.table: self.table[k] = {}
        best = max(self.table.get(nk,{}).get(a,0.0) for a in self.ACTIONS)
        old  = self.table[k].get(action,0.0)
        self.table[k][action] = old + self.alpha*(reward + self.gamma*best - old)

_mab = MAB()
_qt  = QTable()

# ─── Environment ─────────────────────────────────────────────────
class Env:
    def __init__(self):
        self.task=""; self.run_id=""; self.algo_idx=0
        self.attempts: List[Dict]=[]
        self.done=False; self.started=False

    def reset(self, task: str):
        self.task=task; self.run_id=uuid.uuid4().hex[:8]
        self.algo_idx=0; self.attempts=[]; self.done=False; self.started=True

    def obs(self) -> dict:
        if not self.started:
            return {"task":"","attempt_number":0,"algorithm":"","level":"",
                    "action_space":["reason","calculate","retry","change_strategy","submit"],
                    "previous_score":None,"previous_response":None,"context":{}}
        key  = ALGO_ORDER[min(self.algo_idx,2)]
        algo = ALGORITHMS[key]
        prev = self.attempts[-1] if self.attempts else None
        return {
            "task":              self.task,
            "attempt_number":    self.algo_idx + 1,
            "algorithm":         algo["name"],
            "level":             algo["level"],
            "action_space":      ["reason","calculate","retry","change_strategy","submit"],
            "previous_score":    prev["score"] if prev else None,
            "previous_response": prev["response"][:200] if prev else None,
            "context":           {"algo_key": key, "run_id": self.run_id,
                                  "attempts_used": len(self.attempts)},
        }

_env = Env()

# ─── LLM via OpenRouter ──────────────────────────────────────────
def call_llm(system: str, user: str, max_tokens: int = 800) -> str:
    try:
        resp = req_lib.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://metamind.env",
                "X-Title":       "MetaMind OpenEnv",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
            timeout=60,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"

def evaluate(response: str, task: str) -> Dict[str, Any]:
    prompt = (
        f"Rate this AI response from 0.00 to 1.00.\n"
        f"Task: \"{task}\"\n"
        f"Response: \"{response[:500]}\"\n\n"
        f"Scale: 0.00-0.40 wrong/vague | 0.40-0.60 partial | "
        f"0.60-0.75 correct-brief | 0.75-0.90 good | 0.90-1.00 excellent\n\n"
        f"Reply ONLY with valid JSON, no markdown:\n"
        f"{{\"score\": 0.XX, \"reason\": \"one sentence\"}}"
    )
    raw = call_llm("You are a strict evaluator. Output only valid JSON.", prompt, 100)
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        d = json.loads(cleaned)
        return {"score": min(1.0, max(0.0, float(d["score"]))), "reason": str(d.get("reason",""))}
    except Exception:
        m = re.search(r'[\d.]+', raw)
        return {"score": float(m.group()) if m else 0.5, "reason": "parsed"}

# ─── Step logic ──────────────────────────────────────────────────
def do_step() -> Dict[str, Any]:
    key  = ALGO_ORDER[_env.algo_idx]
    algo = ALGORITHMS[key]
    prev = _env.attempts[-1] if _env.attempts else None
    n    = _env.algo_idx + 1

    # Build prompt
    arm, q_action = None, None
    user_msg = _env.task
    if key == "multi_armed_bandit":
        arm = _mab.select()
        if prev:
            user_msg = (
                f"Task: {_env.task}\n\n"
                f"Previous attempt ({prev['algorithm']}) scored "
                f"{prev['score']*100:.0f}% — response: \"{prev['response'][:120]}...\"\n\n"
                f"Improve using '{arm.replace('_',' ')}' strategy."
            )
    elif key == "q_learning":
        q_action = _qt.select(n, prev["score"] if prev else None)
        all_prev = "\n".join(
            f"- {a['algorithm']} scored {a['score']*100:.0f}%: \"{a['response'][:100]}...\""
            for a in _env.attempts
        )
        user_msg = (
            f"Task: {_env.task}\n\n"
            f"Previous failed attempts:\n{all_prev}\n\n"
            f"Q-table selected action: '{q_action.replace('_',' ')}'. "
            f"Produce your optimal answer."
        )

    # Generate
    response = call_llm(algo["system"], user_msg)

    # Evaluate
    ev     = evaluate(response, _env.task)
    score  = ev["score"]
    reason = ev["reason"]
    passed = score >= algo["threshold"]

    # Reward
    base      = algo["r_pass"] if passed else algo["r_fail"]
    step_pen  = n * algo["pen"]
    adapt_bon = (score - prev["score"]) * 0.7 if prev and score > prev["score"] else 0.0
    total_r   = round(base - step_pen + adapt_bon, 4)

    # Update agents
    if key == "multi_armed_bandit" and arm:
        _mab.update(arm, total_r)
    elif key == "q_learning" and q_action:
        _qt.update(n, prev["score"] if prev else None, q_action, total_r, score)

    # Record
    record = {
        "algorithm": algo["name"], "level": algo["level"],
        "algo_key": key, "response": response,
        "score": score, "reward": total_r, "passed": passed,
        "reason": reason, "arm": arm, "q_action": q_action,
    }
    _env.attempts.append(record)

    # Advance state
    if passed or _env.algo_idx >= 2:
        _env.done = True
    else:
        _env.algo_idx += 1

    return {
        "observation": _env.obs(),
        "reward": total_r,
        "done":   _env.done,
        "info": {
            "algorithm":       algo["name"],
            "level":           algo["level"],
            "algo_key":        key,
            "response":        response,
            "score":           score,
            "passed":          passed,
            "threshold":       algo["threshold"],
            "reason":          reason,
            "base_reward":     base,
            "step_penalty":    step_pen,
            "adaptation_bonus":adapt_bon,
            "arm":             arm,
            "q_action":        q_action,
            "run_id":          _env.run_id,
        },
    }

# ─────────────────────────────────────────────────────────────────
# OpenEnv API Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "MetaMind",
        "version": "1.0.0",
        "description": "Self-Improving AI: Contextual Bandit → MAB → Q-Learning",
        "endpoints": ["/reset (POST)", "/step (POST)", "/state (GET)"],
    }


@app.post("/reset")
async def reset(request: Request):
    """
    OpenEnv /reset endpoint.
    Accepts: {"task": "..."} or any body with a task field.
    Returns: {"observation": {...}, "info": {...}}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract task — be very permissive
    task = (
        body.get("task") or
        body.get("query") or
        body.get("prompt") or
        body.get("input") or
        "default task"
    )
    if not isinstance(task, str):
        task = str(task)
    task = task.strip() or "default task"

    _env.reset(task)

    return JSONResponse({
        "observation": _env.obs(),
        "info": {
            "run_id":      _env.run_id,
            "task":        task,
            "algo_order":  ALGO_ORDER,
            "algorithms":  {k: {"name": v["name"], "level": v["level"], "threshold": v["threshold"]}
                            for k, v in ALGORITHMS.items()},
        },
    })


@app.post("/step")
async def step(request: Request):
    """
    OpenEnv /step endpoint.
    Accepts: {"action": ...} (action is ignored — algo auto-selected)
    Returns: {"observation": {...}, "reward": float, "done": bool, "info": {...}}
    """
    if not _env.started:
        # Auto-init with empty task if checker calls step before reset
        _env.reset("default task")

    if _env.done:
        return JSONResponse({
            "observation": _env.obs(),
            "reward": 0.0,
            "done": True,
            "info": {"message": "Episode done. Call /reset to start a new episode."},
        })

    result = do_step()
    return JSONResponse(result)


@app.get("/state")
def state():
    """OpenEnv /state endpoint — current environment state."""
    return JSONResponse({
        "run_id":        _env.run_id,
        "task":          _env.task,
        "algo_idx":      _env.algo_idx,
        "current_algo":  ALGO_ORDER[min(_env.algo_idx, 2)] if _env.started else None,
        "attempts_made": len(_env.attempts),
        "done":          _env.done,
        "started":       _env.started,
        "observation":   _env.obs(),
    })


@app.post("/run")
async def run(request: Request):
    """
    Convenience: run full pipeline in one call.
    Body: {"task": "..."}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task = str(body.get("task", "default task")).strip() or "default task"
    _env.reset(task)

    steps = []
    for _ in range(3):
        if _env.done:
            break
        result = do_step()
        steps.append(result)

    total_r    = sum(s["reward"] for s in steps)
    first_sc   = steps[0]["info"]["score"] if steps else 0
    last_sc    = steps[-1]["info"]["score"] if steps else 0
    adapt_gain = round(max(0.0, last_sc - first_sc), 4)

    return JSONResponse({
        "run_id":            _env.run_id,
        "task":              task,
        "steps":             steps,
        "total_reward":      round(total_r, 4),
        "winning_algorithm": steps[-1]["info"]["algorithm"] if steps else None,
        "final_score":       last_sc,
        "adaptation_gain":   adapt_gain,
        "attempts_used":     len(steps),
        "success":           steps[-1]["done"] and steps[-1]["info"]["passed"] if steps else False,
    })
