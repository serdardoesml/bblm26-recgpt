from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from .common import get_base_dir, print0, setup_distributed
from .dataloader import batch_iterator, count_dataset_tokens
from .model import RecGPTConfig, RecGPTForCausalLM
from .optimizer import SingleDeviceNorMuonWithAuxAdam


@dataclass
class TrainConfig:
    model_config: RecGPTConfig = field(default_factory=RecGPTConfig)
    dataset: str = "bblm10M.parquet"
    tokenizer: str = "bblm10M-bpe"
    run_name: str = "recgpt"
    seed: int = 0
    data_seed: int = 0

    microbatch_tok: int = 16384 # Tokens per microbatch (before grad accumulation) per gpu
    total_batch_tok: int = 16384 # Tokens per gradient step. Must be a multiple of microbatch_tok * gpu count.
    sequence_len: int = 512
    epochs: int = 1

    # Token limit (per epoch), -1 means use the entire dataset.
    # Note: Don't use this for multi epoch training on a subset, as each epoch will see a different subset of the data due to shuffling.
    max_tokens: int = -1 

    lr_embed: float = 0.007
    lr_block: float = 0.02
    min_lr_embed: float = 0.0
    min_lr_block: float = 0.0
    wd_adam: float = 0.005
    wd_muon: float = 0.1
    warmup_steps: int = 50
    cooldown_steps: int = 400
    max_grad_norm: float = 2.0

    torch_compile: bool = True
    use_wandb: bool = False
    wandb_project: str = "bblm26-recgpt"
    log_every: int = 10
    save_dir: str = "models"


def unwrap_model(model: torch.nn.Module) -> RecGPTForCausalLM:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"): # torch.compile
        model = model._orig_mod
    return model


def build_optimizer(model: RecGPTForCausalLM, cfg: TrainConfig):
    adam_params = []
    muon_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name or "e_to_h" in name or "h_to_e" in name:
            adam_params.append(p)
        else:
            muon_params.append(p)

    return SingleDeviceNorMuonWithAuxAdam(
        [
            {"params": adam_params, "lr": cfg.lr_embed, "use_muon": False, "weight_decay": cfg.wd_adam},
            {"params": muon_params, "lr": cfg.lr_block, "use_muon": True, "weight_decay": cfg.wd_muon},
        ]
    )


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    cooldown_steps: int,
    total_steps: int,
    min_lrs: list[float],
):
    # Linear warmup, constant phase, then linear cooldown to min_lrs.
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    min_factors = [min_lr / lr if lr > 0 else 1.0 for min_lr, lr in zip(min_lrs, base_lrs, strict=True)]
    warmup_steps = max(0, warmup_steps)
    cooldown_steps = max(0, cooldown_steps)
    total_steps = max(1, total_steps)
    constant_steps = max(0, total_steps - warmup_steps - cooldown_steps)

    def build_lambda(min_factor: float):
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return 1e-8 + (1.0 - 1e-8) * (step / warmup_steps)
            if step < warmup_steps + constant_steps:
                return 1.0
            if cooldown_steps == 0:
                return min_factor
            progress = min(1.0, (step - warmup_steps - constant_steps) / cooldown_steps)
            return (1.0 - progress) * (1.0 - min_factor) + min_factor
        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [build_lambda(f) for f in min_factors])


def config_to_json(cfg: TrainConfig) -> dict:
    out = {k: v for k, v in cfg.__dict__.items() if k != "model_config"}
    out["model_config"] = cfg.model_config.to_dict()
    return out


def train(cfg: TrainConfig):
    device, rank, local_rank, world_size, ddp = setup_distributed()
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    torch.set_float32_matmul_precision("high")

    root = get_base_dir()
    parquet_path = root / "data" / "tokenized" / cfg.dataset
    tokenizer_path = root / "tokenizers" / cfg.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    cfg.model_config.vocab_size = len(tokenizer)
    cfg.model_config.pad_token_id = tokenizer.pad_token_id
    cfg.model_config.max_position_embeddings = cfg.sequence_len

    raw_model = RecGPTForCausalLM(cfg.model_config).to(device)
    model: torch.nn.Module = torch.compile(raw_model) if cfg.torch_compile else raw_model
    if ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = build_optimizer(raw_model, cfg)
    dataset_tokens = count_dataset_tokens(parquet_path)
    epoch_tokens = dataset_tokens if cfg.max_tokens < 0 else min(cfg.max_tokens, dataset_tokens)
    target_tokens = epoch_tokens * cfg.epochs
    global_microbatch_tok = cfg.microbatch_tok * world_size
    if cfg.total_batch_tok < global_microbatch_tok or cfg.total_batch_tok % global_microbatch_tok != 0:
        raise ValueError("total_batch_tok must be a multiple of microbatch_tok * world_size.")
    grad_acc = cfg.total_batch_tok // global_microbatch_tok
    total_microbatches = max(1, math.ceil(target_tokens / global_microbatch_tok))
    total_steps = max(1, math.ceil(total_microbatches / grad_acc))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        cfg.warmup_steps,
        cfg.cooldown_steps,
        total_steps,
        [cfg.min_lr_embed, cfg.min_lr_block],
    )

    wandb_run = None
    if cfg.use_wandb and rank == 0:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=config_to_json(cfg))

    print0(
        f"training {cfg.run_name} | tokens {target_tokens} | steps {total_steps} | "
        f"microbatch_tok {cfg.microbatch_tok} | total_batch_tok {cfg.total_batch_tok} | "
        f"grad_acc {grad_acc} | world_size {world_size}",
        rank=rank,
    )

    step = 0
    micro_step = 0
    tokens_seen = 0
    loss_accum = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(cfg.epochs if cfg.max_tokens < 0 else 10**12):
        iterator = batch_iterator(
            parquet_path,
            tokens_per_batch=cfg.microbatch_tok,
            max_sl=cfg.sequence_len,
            pad_token_id=cfg.model_config.pad_token_id,
            device=device,
            seed=cfg.data_seed + epoch,
            rank=rank,
            world_size=world_size,
        )
        for input_ids, labels, segment_ids, position_ids in iterator:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, segment_ids=segment_ids, position_ids=position_ids, labels=labels)
                loss = out.loss / grad_acc

            loss.backward()
            loss_accum += float(loss.detach()) * grad_acc
            tokens_seen += global_microbatch_tok
            micro_step += 1

            if micro_step % grad_acc != 0:
                continue

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                metrics = {
                    "loss": loss_accum,
                    "lr_embed": optimizer.param_groups[0]["lr"],
                    "lr_block": optimizer.param_groups[1]["lr"],
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_seen / elapsed,
                }
                print0(f"step {step:05d} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()), rank=rank)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=tokens_seen)

            step += 1
            loss_accum = 0.0
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if ddp:
        torch.distributed.barrier()
    if rank == 0:
        out_dir = root / cfg.save_dir / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(config_to_json(cfg), f, indent=2)
        print(f"saved model to {out_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    if ddp:
        torch.distributed.destroy_process_group()
