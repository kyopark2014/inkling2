#!/bin/sh
# SageMaker starts the container with a trailing "serve" argument; ignore it.
set -eu
exec python3 /opt/program/sagemaker_entrypoint.py
