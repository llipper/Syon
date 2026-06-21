"""Stub de servidor gRPC para inferência Syon."""

from __future__ import annotations

from typing import Any


class InferenceServicer:
    """Stub do serviço gRPC Inference."""

    def __init__(self, engine: Any = None):
        self.engine = engine

    def Generate(self, request: Any, context: Any) -> Any:
        raise NotImplementedError("gRPC Generate() — implementar com protobufs em api/grpc/")

    def AnalyzeSecurity(self, request: Any, context: Any) -> Any:
        raise NotImplementedError("gRPC AnalyzeSecurity() — implementar com protobufs")

    def StreamGenerate(self, request: Any, context: Any):
        raise NotImplementedError("gRPC StreamGenerate() — implementar com protobufs")
        yield None


def serve(port: int = 50051) -> None:
    """Inicia servidor gRPC (stub)."""
    raise NotImplementedError(
        "gRPC server é um stub. Defina protos em api/grpc/protos/ e gere stubs."
    )