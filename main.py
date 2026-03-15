"""
FastAPI application entry point for SupportMailAgent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.routes import emails
from src.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="SupportMailAgent",
    description="AI-powered customer support email agent",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(emails.router, prefix="/emails", tags=["emails"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Serve static files (frontend)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
