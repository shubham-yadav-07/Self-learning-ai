"""
MetaMind — OpenEnv Self-Improvement Environment
=================================================
Fully OpenEnv-spec-compliant HTTP server.

OpenEnv Observation base class requires: done, reward, metadata
StepResult shape: { observation, reward, done }

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os, json, math, random, re, time, uuid
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests as http

load_dotenv()

OPENROUTER_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="MetaMind", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# OpenEnv base Observation shape (from RFC-002)
# Every observation MUST contain: done, reward, metadata
# Plus our custom fields.
# ─────────────────────────────────────────────────────────────────

def make_observation(
    *,
    task: str = "",
    attempt_number: int = 0,
    algorithm: str = "",
    level: str = "",
    previous_score: Optional[float] = None,
    done: bool = False,
    reward: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Returns an observation dict that conforms to OpenEnv Observation base class.
    Required base fields: done, reward, metadata
    Custom fields: task, attempt_number, algorithm, level, previous_score
    """
    return {
        # ── OpenEnv required base fields ──────────────────────
        "done":     done,
        "reward":   reward,
        "metadata": metadata or {},
        # ── MetaMind custom fields ────────────────────────────
        "task":           task,
        "attempt_number": attempt_number,
        "algorithm":      algorithm,
        "level":          level,
        "previous_score": previous_score,
        "action_space":   ["reason", "calculate", "retry", "change_strategy", "submit"],
    }


def make_step_result(
    observation: Dict[str, Any],
    reward: float,
    done: bool,
    info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Returns the top-level HTTP response body for /reset and /step.
    Shape: { observation, reward, done }
    """
    return {
        "observation": observation,
        "reward":      reward,
        "done":        done,
        # info is extra / optional but we include it for transparency
        "info":        info or {},
    }

# ─────────────────────────────────────────────────────────────────
# Algorithm config
# ─────────────────────────────────────────────────────────────────

ALGO_ORDER = ["contextual_bandit", "multi_armed_bandit", "q_learning"]

ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "contextual_bandit": {
        "name":      "Contextual Bandit",
        "level":     "easy",
        "threshold": 0.60,
        "r_pass":    1.0,
        "r_fail":   -0.5,
        "pen":       0.05,
        "system": (
            "You are a Level-1 Contextual Bandit AI. "
            "Policy: pure greedy exploitation. "
            "Give the most direct, concise answer in 2-3 sentences only."
        ),
    },
    "multi_armed_bandit": {
        "name":      "Multi-Armed Bandit",
        "level":     "medium",
        "threshold": 0.75,
        "r_pass":    1.5,
        "r_fail":   -0.3,
        "pen":       0.03,
        "system": (
            "You are a Level-2 Multi-Armed Bandit AI using UCB1 exploration. "
            "The previous attempt was insufficient. "
            "Briefly state your chosen strategy arm, then give a better answer in 4-6 sentences."
        ),
    },
    "q_learning": {
        "name":      "Q-Learning",
        "level":     "hard",
        "threshold": 0.85,
        "r_pass":    2.0,
        "r_fail":   -0.1,
        "pen":       0.01,
        "system": (
            "You are a Level-3 Q-Learning AI optimising cumulative reward. "
            "Previous attempts scored below threshold. "
            "Produce your absolute best, well-structured, comprehensive answer."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────
# Agent internals
# ─────────────────────────────────────────────────────────────────

class MAB:
    ARMS = ["direct", "step_by_step", "analogy", "structured", "first_principles"]
    C    = math.sqrt(2)

    def __init__(self):
        self.counts = {a: 0   for a in self.ARMS}
        self.values = {a: 0.0 for a in self.ARMS}
        self.total  = 0

    def select(self) -> str:
        self.total += 1
        for a in self.ARMS:
            if self.counts[a] == 0:
                return a
        return max(self.ARMS, key=lambda a:
            self.values[a] + self.C * math.sqrt(math.log(self.total) / self.counts[a]))

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class QTable:
    ACTIONS = ["comprehensive", "structured", "examples", "concise", "socratic"]

    def __init__(self):
        self.table: Dict[str, Dict[str, float]] = {}
        self.alpha = 0.15
        self.gamma = 0.90
        self.eps   = 0.10

    def _k(self, i: int, s: Optional[float]) -> str:
        return f"a{i}_s{'none' if s is None else int(s * 10)}"

    def select(self, i: int, s: Optional[float]) -> str:
        if random.random() < self.eps:
            return random.choice(self.ACTIONS)
        k = self._k(i, s)
        return max(self.ACTIONS, key=lambda a: self.table.get(k, {}).get(a, 0.0))

    def update(self, i: int, s: Optional[float], action: str, reward: float, ns: float):
        k  = self._k(i, s)
        nk = self._k(i + 1, ns)
        if k not in self.table:
            self.table[k] = {}
        best = max(self.table.get(nk, {}).get(a, 0.0) for a in self.ACTIONS)
        old  = self.table[k].get(action, 0.0)
        self.table[k][action] = old + self.alpha * (reward + self.gamma * best - old)


_mab = MAB()
_qt  = QTable()

# ─────────────────────────────────────────────────────────────────
# Environment state
# ─────────────────────────────────────────────────────────────────

class Env:
    def __init__(self):
        self.task     = "default task"
        self.run_id   = ""
        self.algo_idx = 0
        self.attempts: List[Dict] = []
        self.done     = False
        self.started  = False

    def reset(self, task: str):
        self.task     = task or "default task"
        self.run_id   = uuid.uuid4().hex[:8]
        self.algo_idx = 0
        self.attempts = []
        self.done     = False
        self.started  = True

    def current_obs(self, done: bool = False, reward: Optional[float] = None) -> Dict:
        key  = ALGO_ORDER[min(self.algo_idx, 2)]
        algo = ALGORITHMS[key]
        prev = self.attempts[-1] if self.attempts else None
        return make_observation(
            task           = self.task,
            attempt_number = self.algo_idx + 1,
            algorithm      = algo["name"],
            level          = algo["level"],
            previous_score = prev["score"] if prev else None,
            done           = done,
            reward         = reward,
            metadata       = {
                "run_id":    self.run_id,
                "algo_key":  key,
                "threshold": algo["threshold"],
            },
        )


_env = Env()

# ─────────────────────────────────────────────────────────────────
# LLM helpers (OpenRouter)
# ─────────────────────────────────────────────────────────────────

def llm(system: str, user: str, max_tokens: int = 800) -> str:
    try:
        r = http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://metamind.env",
                "X-Title":       "MetaMind",
            },
            json={
                "model":      MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
            timeout=60,
        )
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def evaluate(response: str, task: str) -> Dict[str, Any]:
    prompt = (
        f"Rate this AI response 0.00 to 1.00.\n"
        f"Task: \"{task}\"\n"
        f"Response: \"{response[:500]}\"\n\n"
        f"0.00-0.40 wrong  0.40-0.60 partial  0.60-0.75 brief  "
        f"0.75-0.90 good  0.90-1.00 excellent\n\n"
        f"Reply ONLY with valid JSON, no markdown:\n"
        f"{{\"score\": 0.XX, \"reason\": \"one sentence\"}}"
    )
    raw = llm("You are a strict evaluator. Output only valid JSON.", prompt, 80)
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        d = json.loads(cleaned)
        return {"score": min(1.0, max(0.0, float(d["score"]))), "reason": str(d.get("reason", ""))}
    except Exception:
        m = re.search(r"[\d.]+", raw)
        return {"score": float(m.group()) if m else 0.5, "reason": "auto-parsed"}

# ─────────────────────────────────────────────────────────────────
# Core step logic
# ─────────────────────────────────────────────────────────────────

def _do_step() -> Dict[str, Any]:
    key  = ALGO_ORDER[_env.algo_idx]
    algo = ALGORITHMS[key]
    prev = _env.attempts[-1] if _env.attempts else None
    n    = _env.algo_idx + 1

    # Build user message
    arm, q_action = None, None
    user_msg = _env.task

    if key == "multi_armed_bandit":
        arm = _mab.select()
        if prev:
            user_msg = (
                f"Task: {_env.task}\n\n"
                f"Previous attempt scored {prev['score']*100:.0f}% — "
                f"response: \"{prev['response'][:100]}...\"\n\n"
                f"Improve using '{arm.replace('_', ' ')}' strategy."
            )
    elif key == "q_learning":
        q_action = _qt.select(n, prev["score"] if prev else None)
        prev_summary = "\n".join(
            f"- {a['algorithm']} scored {a['score']*100:.0f}%: \"{a['response'][:80]}...\""
            for a in _env.attempts
        )
        user_msg = (
            f"Task: {_env.task}\n\n"
            f"Previous failed attempts:\n{prev_summary}\n\n"
            f"Q-table action: '{q_action.replace('_', ' ')}'. Produce your optimal answer."
        )

    # Generate response
    response = llm(algo["system"], user_msg)

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
    _env.attempts.append({
        "algorithm": algo["name"], "level": algo["level"], "algo_key": key,
        "response": response, "score": score, "reward": total_r, "passed": passed,
        "reason": reason, "arm": arm, "q_action": q_action,
    })

    # Advance
    if passed or _env.algo_idx >= 2:
        _env.done = True
    else:
        _env.algo_idx += 1

    # Build observation (reward and done go INSIDE observation too — per OpenEnv spec)
    obs = _env.current_obs(done=_env.done, reward=total_r)

    return make_step_result(
        observation=obs,
        reward=total_r,
        done=_env.done,
        info={
            "algorithm":        algo["name"],
            "level":            algo["level"],
            "response":         response,
            "score":            score,
            "passed":           passed,
            "threshold":        algo["threshold"],
            "reason":           reason,
            "base_reward":      base,
            "step_penalty":     step_pen,
            "adaptation_bonus": adapt_bon,
            "arm":              arm,
            "q_action":         q_action,
        },
    )

# ─────────────────────────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "MetaMind",
        "version": "1.0.0",
        "description": "Self-Improving AI: Contextual Bandit → MAB → Q-Learning",
        "endpoints": {
            "POST /reset": "Reset environment",
            "POST /step":  "Execute one step",
            "GET  /state": "Get current state",
            "POST /run":   "Full pipeline in one call",
        },
    }


@app.post("/reset")
async def reset(request: Request):
    """
    OpenEnv /reset endpoint.
    Accepts any JSON body. Extracts 'task' if present.
    Returns StepResult shape: { observation, reward, done }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task = str(
        body.get("task") or body.get("query") or body.get("prompt") or "default task"
    ).strip() or "default task"

    _env.reset(task)

    # Initial observation — reward=None, done=False (not stepped yet)
    obs = make_observation(
        task           = _env.task,
        attempt_number = 1,
        algorithm      = ALGORITHMS["contextual_bandit"]["name"],
        level          = ALGORITHMS["contextual_bandit"]["level"],
        previous_score = None,
        done           = False,
        reward         = None,
        metadata       = {
            "run_id":    _env.run_id,
            "algo_key":  "contextual_bandit",
            "threshold": ALGORITHMS["contextual_bandit"]["threshold"],
            "message":   "Environment reset. Call POST /step to begin.",
        },
    )

    return JSONResponse(make_step_result(
        observation=obs,
        reward=0.0,
        done=False,
        info={"run_id": _env.run_id, "task": _env.task},
    ))


@app.post("/step")
async def step(request: Request):
    """
    OpenEnv /step endpoint.
    Accepts any JSON body (action is auto-selected by algo).
    Returns StepResult shape: { observation, reward, done }
    """
    # Auto-init if never reset
    if not _env.started:
        _env.reset("default task")

    if _env.done:
        obs = _env.current_obs(done=True, reward=0.0)
        return JSONResponse(make_step_result(
            observation=obs, reward=0.0, done=True,
            info={"message": "Episode done. Call POST /reset to start a new episode."},
        ))

    result = _do_step()
    return JSONResponse(result)


@app.get("/state")
def state():
    """OpenEnv /state endpoint."""
    key  = ALGO_ORDER[min(_env.algo_idx, 2)]
    algo = ALGORITHMS[key]
    return JSONResponse({
        "run_id":        _env.run_id,
        "task":          _env.task,
        "algo_idx":      _env.algo_idx,
        "current_algo":  algo["name"],
        "current_level": algo["level"],
        "attempts_made": len(_env.attempts),
        "done":          _env.done,
        "started":       _env.started,
        "observation":   _env.current_obs(done=_env.done),
    })


@app.post("/run")
async def run(request: Request):
    """Convenience: full pipeline in one HTTP call."""
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
        result = _do_step()
        steps.append(result)

    total_r    = sum(s["reward"] for s in steps)
    first_sc   = steps[0]["info"]["score"] if steps else 0.0
    last_sc    = steps[-1]["info"]["score"] if steps else 0.0
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
