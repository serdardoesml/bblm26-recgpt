import argparse

from main.model import RecGPTConfig
from main.train import TrainConfig, train


def parse_args() -> TrainConfig:
    model_defaults = RecGPTConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bblm10M.parquet")
    parser.add_argument("--tokenizer", default="bblm10M-bpe")
    parser.add_argument("--run-name", default=TrainConfig.run_name)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--max-tokens", type=int, default=-1)
    parser.add_argument("--microbatch-tok", type=int, default=TrainConfig.microbatch_tok)
    parser.add_argument("--total-batch-tok", type=int, default=TrainConfig.total_batch_tok)
    parser.add_argument("--sequence-len", type=int, default=TrainConfig.sequence_len)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=model_defaults.hidden_size)
    parser.add_argument("--embedding-size", type=int, default=model_defaults.embedding_size)
    parser.add_argument("--head-dim", type=int, default=model_defaults.head_dim)
    parser.add_argument("--intermediate-size", type=int, default=model_defaults.intermediate_size)
    parser.add_argument("--recursive-depth", type=int, default=model_defaults.recursive_depth)
    parser.add_argument("--tie-word-embeddings", type=bool, default=model_defaults.tie_word_embeddings)
    args = parser.parse_args()

    return TrainConfig(
        model_config=RecGPTConfig(
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            head_dim=args.head_dim,
            intermediate_size=args.intermediate_size,
            recursive_depth=args.recursive_depth,
            tie_word_embeddings=args.tie_word_embeddings,
        ),
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
