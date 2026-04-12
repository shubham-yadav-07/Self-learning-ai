"""
app.py — root HF Spaces entry point
"""
from main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
