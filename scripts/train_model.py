import argparse

from main.train import TrainConfig, train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bblm10M.parquet")
    parser.add_argument("--tokenizer", default="bblm10M-bpe")
    parser.add_argument("--run-name", default="recgpt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=-1)
    parser.add_argument("--microbatch-tok", type=int, default=16384)
    parser.add_argument("--total-batch-tok", type=int, default=16384)
    parser.add_argument("--sequence-len", type=int, default=512)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        run_name=args.run_name,
        epochs=args.epochs,
        max_tokens=args.max_tokens,
        microbatch_tok=args.microbatch_tok,
        total_batch_tok=args.total_batch_tok,
        sequence_len=args.sequence_len,
        torch_compile=not args.no_compile,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    train(parse_args())
