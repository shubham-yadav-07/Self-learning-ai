"""
app.py — Hugging Face Spaces entry point
=========================================
HF Spaces looks for app.py at repo root.
This file simply imports and exposes the FastAPI app from main.py.
 
HF Spaces with Docker SDK uses the Dockerfile directly,
but app.py is kept here for sdk: gradio / sdk: streamlit fallback
and for openenv validate compatibility.
"""
 
# Re-export the FastAPI app so any runner that imports app.py gets it
from main import app  # noqa: F401
 
# ── If run directly, start uvicorn ──────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 7860))  # HF Spaces default port
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)