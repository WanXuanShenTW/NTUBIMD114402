#!/bin/bash
export FLASK_APP=run.py
export PYTHONUNBUFFERED=1
flask run --host=0.0.0.0 --port=5000