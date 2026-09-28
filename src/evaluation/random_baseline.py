import argparse
import json
from src.evaluation.synthetic import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--source', choices=['natural', 'reverse'], default='natural')
    p.add_argument('--target-remaining', type=int, default=98)
    args = p.parse_args()
    print(json.dumps(evaluate_model(None, source=args.source, target_remaining=args.target_remaining,
                                   games=args.games, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
