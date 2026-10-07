import argparse
import os

from data import (count_matrix, dirichlet_partition, load_cifar10, partition_path,
                  print_counts, save_partition)


def parse_alpha(s):
    return None if s == "iid" else float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--alphas", type=parse_alpha, nargs="+", default=[0.1, 0.5, 5.0, None])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_size", type=int, default=256)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="./partitions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _, labels = load_cifar10(args.data_root)

    #partitions are made once here and only loaded everywhere else,
    # so every method sees exactly the same clients
    for alpha in args.alphas:
        parts = dirichlet_partition(labels, args.n_clients, alpha, args.seed, args.min_size)
        path = partition_path(args.out_dir, args.n_clients, alpha, args.seed)
        save_partition(path, parts, alpha, args.seed)
        print_counts(count_matrix(labels, parts), title=f"{path}")


if __name__ == "__main__":
    main()
