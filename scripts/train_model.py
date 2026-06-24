import argparse

from main.model import RecGPTConfig
from main.train import TrainConfig, train


def parse_args() -> TrainConfig:
    train_defaults = TrainConfig()
    model_defaults = RecGPTConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=train_defaults.dataset)
    parser.add_argument("--tokenizer", default=train_defaults.tokenizer)
    parser.add_argument("--run-name", default=train_defaults.run_name)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--data-seed", type=int, default=train_defaults.data_seed)
    parser.add_argument("--microbatch-tok", type=int, default=train_defaults.microbatch_tok)
    parser.add_argument("--total-batch-tok", type=int, default=train_defaults.total_batch_tok)
    parser.add_argument("--sequence-len", type=int, default=train_defaults.sequence_len)
    parser.add_argument("--epochs", type=int, default=train_defaults.epochs)
    parser.add_argument("--max-tokens", type=int, default=train_defaults.max_tokens)
    parser.add_argument("--lr-embed", type=float, default=train_defaults.lr_embed)
    parser.add_argument("--lr-block", type=float, default=train_defaults.lr_block)
    parser.add_argument("--wd-adam", type=float, default=train_defaults.wd_adam)
    parser.add_argument("--wd-muon", type=float, default=train_defaults.wd_muon)
    parser.add_argument("--warmup-ratio", type=float, default=train_defaults.warmup_ratio)
    parser.add_argument("--cooldown-ratio", type=float, default=train_defaults.cooldown_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=train_defaults.max_grad_norm)
    parser.add_argument("--nl-mult", type=float, default=train_defaults.nl_mult)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=train_defaults.wandb_project)
    parser.add_argument("--vocab-size", type=int, default=model_defaults.vocab_size)
    parser.add_argument("--hidden-size", type=int, default=model_defaults.hidden_size)
    parser.add_argument("--embedding-size", type=int, default=model_defaults.embedding_size)
    parser.add_argument("--head-dim", type=int, default=model_defaults.head_dim)
    parser.add_argument("--intermediate-size", type=int, default=model_defaults.intermediate_size)
    parser.add_argument("--recursive-depth", type=int, default=model_defaults.recursive_depth)
    parser.add_argument("--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=model_defaults.tie_word_embeddings)
    args = parser.parse_args()

    return TrainConfig(
        model_config=RecGPTConfig(
            vocab_size=args.vocab_size,
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
        seed=args.seed,
        data_seed=args.data_seed,
        microbatch_tok=args.microbatch_tok,
        total_batch_tok=args.total_batch_tok,
        sequence_len=args.sequence_len,
        epochs=args.epochs,
        max_tokens=args.max_tokens,
        lr_embed=args.lr_embed,
        lr_block=args.lr_block,
        wd_adam=args.wd_adam,
        wd_muon=args.wd_muon,
        warmup_ratio=args.warmup_ratio,
        cooldown_ratio=args.cooldown_ratio,
        max_grad_norm=args.max_grad_norm,
        nl_mult=args.nl_mult,
        torch_compile=not args.no_compile,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    train(parse_args())
