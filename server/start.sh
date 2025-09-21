#!/bin/bash
export PYTHONUNBUFFERED=1
uvicorn run:app --host=0.0.0.0 --port=5000