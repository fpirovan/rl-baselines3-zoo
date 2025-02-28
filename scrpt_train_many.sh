#!/usr/bin/env bash

proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export PYTHONPATH="${PYTHONPATH}:${proj_dir}"
cmd=$(python args_train_many.py --seed 1085 "$@")

cd "$proj_dir"
eval "$cmd"
