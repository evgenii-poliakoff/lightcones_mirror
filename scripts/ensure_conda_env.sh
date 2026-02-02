#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

set -euo pipefail

conda_env_dir="${script_dir}"/../conda-env

is_force='false'

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            is_force='true'
            shift
            ;;
        *)
            echo ERROR "Unknown option: '$1'"
            echo "Usage: $(basename $0) [-f|--force]"
            exit 1
            ;;
    esac
done

if [[ -d "${conda_env_dir}" ]] ; then
    if [[ "${is_force}" == 'true' ]] ; then
        echo "INFO Removing '${conda_env_dir}' conda environment directory"
        rm -rf "${conda_env_dir}"
    else
        echo "INFO conda environment already exists at '${conda_env_dir}'. Skipping creating"
        exit 0
    fi
fi

echo "INFO Creating conda environment in '${conda_env_dir}' directory"

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

echo "INFO: Downloading required conda packages..."
# Цикл будет перезапускать команду, пока всё не скачается на 100%
until conda create -y --prefix "${conda_env_dir}" python=3.12 --download-only; do
  echo "Connection lost. Retrying download ..."
  sleep 2
done

conda create -y --prefix "${conda_env_dir}" python=3.12

echo "INFO: Installing packages from cache..."
conda install -y --prefix "${conda_env_dir}" --file "${script_dir}/requirements.txt"

conda run --prefix "${conda_env_dir}" python3 -m ipykernel install --user --name lightcones-env --display-name "lightcones-env"
