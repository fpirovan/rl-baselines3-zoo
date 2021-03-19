from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_ids", type=str, nargs="+", default=["Walker2d-v2"])
    args = parser.parse_args()

    print(" ".join(args.env_ids))
