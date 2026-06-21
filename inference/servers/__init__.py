"""Servidores de inferência Syon."""

from inference.servers.fastapi_server import create_app

__all__ = ["create_app"]