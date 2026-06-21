"""Servidor FastAPI — api.syon.ai."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from syon import __version__
from syon.api.routes import chat, completions, security

app = FastAPI(
    title="Syon API",
    description="API REST para programação e cybersegurança",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SYON_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(completions.router, prefix="/v1", tags=["completions"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(security.router, prefix="/v1", tags=["security"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def run_server() -> None:
    host = os.getenv("SYON_API_HOST", "0.0.0.0")
    port = int(os.getenv("SYON_API_PORT", "8000"))
    uvicorn.run("syon.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()