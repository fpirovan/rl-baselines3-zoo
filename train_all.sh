#!/usr/bin/env bash

proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
environments_dir="${proj_dir}/environments/assets"

env_ids=$(python gen_str_train_all.py --env_ids "$@")

export PYTHONPATH="${PYTHONPATH}:${proj_dir}"
cd "$proj_dir"

for env_id in $env_ids; do
  python train.py --seed 1085 --env "$env_id"
done
