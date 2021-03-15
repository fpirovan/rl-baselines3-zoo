#!/usr/bin/env bash

proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
environments_dir="${proj_dir}/environments/assets"

export PYTHONPATH=$PYTHONPATH:$proj_dir
cd "$proj_dir"

env_ids="Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 Walker2d-v2"
for env_xml in "$environments_dir"/*.xml; do
  env_id=$(basename -- "$env_xml")
  env_id="${env_id%.*}"
  env_id="${env_id}-v1"
  env_ids="${env_ids} ${env_id}"
done

for env_id in $env_ids; do
  python train.py --seed 1085 --env "$env_id"
done
