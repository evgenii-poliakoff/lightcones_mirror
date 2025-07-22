#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
conda_env_dir="${script_dir}"/../conda-env

echo INFO Installing lightcones package

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

conda activate "${conda_env_dir}"
conda install -y --use-local --force-reinstall lightcones