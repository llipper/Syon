"""WebSocket handler stub para streaming de inferência."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("syon.api.websocket")


class WebSocketManager:
    """Gerencia conexões WebSocket para streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Aceita nova conexão WebSocket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected, total=%d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove conexão encerrada."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected, total=%d", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Envia mensagem para todas as conexões ativas."""
        payload = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                self.disconnect(connection)

    async def handle_streaming_inference(self, websocket: WebSocket, prompt: str) -> None:
        """Stub de streaming — envia chunks simulados."""
        await self.connect(websocket)
        try:
            words = prompt.split() or ["Hello"]
            for i, word in enumerate(words):
                await websocket.send_json({
                    "event": "chunk",
                    "index": i,
                    "text": word + " ",
                })
            await websocket.send_json({"event": "complete", "finish_reason": "stop"})
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as exc:
            await websocket.send_json({"event": "error", "detail": str(exc)})
            self.disconnect(websocket)