#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

set -euo pipefail

miniforge3_dir="$HOME/miniforge3"
if [[ -d "${miniforge3_dir}" ]] ; then
    echo "INFO miniforge3 is already installed"
    exit 0
fi

repo_dir="${script_dir}"/repo/miniforge3

if [[ -d "${repo_dir}" ]] ; then
    echo "INFO Local copy of 'miniforge3' prerequisite found in '${repo_dir}' directory"
else
    echo "INFO Downloading 'miniforge3' prerequisite to '${repo_dir}' directory"
    mkdir -p "${repo_dir}"
    pushd "${repo_dir}"
    wget "https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Linux-x86_64.sh"
    popd
fi

echo "INFO Installing miniconda3"

pushd "${repo_dir}"
bash ./Miniforge3-25.3.0-1-Linux-x86_64.sh -b
popd

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

echo "INFO: Downloading conda-build..."
# Цикл будет перезапускать команду, пока всё не скачается на 100%
until conda install -y conda-build --download-only; do
  echo "Connection lost. Retrying download of conda-build..."
  sleep 2
done

echo "INFO: Installing conda-build from cache..."
conda install -y conda-build