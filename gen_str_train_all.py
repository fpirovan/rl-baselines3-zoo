from argparse import ArgumentParser

from environments import register_all_envs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_ids", type=str, nargs="+", default=["Walker2d-v2"])
    args = parser.parse_args()

    new_env_ids = register_all_envs()

    env_ids = args.env_ids
    if "new" in args.env_ids:
        env_ids = [env_id for env_id in env_ids if env_id != "new"]
        env_ids.extend(new_env_ids)

    print(" ".join(env_ids))
