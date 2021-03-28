from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--env_ids", type=str, nargs="+", default=["Walker2d-v2"])
    args = parser.parse_args()

    cmd = ""

    for env_id in args.env_ids:
        cmd += f'python train.py --seed {args.seed} --env {env_id}\n'

    print(cmd)
