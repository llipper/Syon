#!/bin/sh
curl -f http://localhost:${SYON_API_PORT:-8000}/health || exit 1