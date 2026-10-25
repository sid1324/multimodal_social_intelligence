"""Small on-policy RLOO pilot or answer-only SFT control on R3-FDT only.

RLOO loss = mean_i[-(r_i-mean_{j!=i} r_j) sum_t log pi(y_it|x,y_i<t)/L
                    + beta sum_t (exp(log pi_ref-log pi)- (log pi_ref-log pi)-1)/L].
L is a fixed completion cap, not the realized response length. One optimizer
step per group, no replay, temperature=1, top_p=1. Reward does not score rationale.
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from social_rl.common import STORE, append_jsonl, config, read_jsonl, sha256, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import parse_r3, r3_prompt


def save_checkpoint(policy, optimizer, scaler, run, step, data_cursor):
    temp = run / f"checkpoint-{step:05d}.tmp"
    destination = run / f"checkpoint-{step:05d}"
    if temp.exists():
        shutil.rmtree(temp)
    policy.save(temp)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "cursor": data_cursor, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temp / "training_state.pt")
    if destination.exists():
        raise ValueError(f"Refusing to overwrite checkpoint {destination}")
    temp.rename(destination)
    write_json(run / "latest.json", {"checkpoint": str(destination), "step": step, "cursor": data_cursor})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["rloo", "sft"], default="rloo")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-hours", type=float, default=3.5)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = config()
    rows = read_jsonl(args.manifest)
    if not rows or any(r["dataset"] != "r3" or r["split"] != "train" for r in rows):
        raise ValueError("Training requires nonempty R3 training-only manifest")
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"config": cfg, "method": args.method, "manifest_sha256": sha256(args.manifest),
                "examples": len(rows), "videos": len({r['video_id'] for r in rows}), "seed": cfg['seed']}
    if (run / "metadata.json").exists() and json.loads((run / "metadata.json").read_text()) != metadata:
        raise ValueError("Resume settings do not match existing run")
    if (run / "latest.json").exists() and not args.resume:
        raise ValueError("Existing checkpoint; explicitly pass --resume")
    write_json(run / "metadata.json", metadata)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    adapter = json.loads((run / "latest.json").read_text())["checkpoint"] if args.resume else None
    policy = VideoPolicy(adapter=adapter, trainable=True)
    params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    step, cursor = 0, 0
    if adapter:
        state = torch.load(Path(adapter) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, cursor = state["step"], state["cursor"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    start = time.monotonic()
    last_saved = step
    attempted = 0
    while step < args.steps and time.monotonic() - start < args.max_hours * 3600:
        if attempted >= args.steps * 10:
            print("STOP: insufficient reward variation after bounded attempts", flush=True)
            break
        row = rows[order[cursor % len(order)]]
        cursor += 1
        attempted += 1
        permutation = random.sample(range(5), 5)
        target = permutation.index(row["answer"])
        inputs = policy.inputs(r3_prompt(row, permutation), row["video_path"], end_seconds=row.get('clip_duration_seconds'))
        tick = time.monotonic()
        event = {"cursor": cursor, "step_before": step, "id": row["id"], "video_id": row["video_id"],
                 "permutation": permutation, "target": target, "method": args.method}
        optimizer.zero_grad(set_to_none=True)
        if args.method == "sft":
            text = f"Answer: {chr(65 + target)}" + policy.processor.tokenizer.eos_token
            completion = policy.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(policy.device)
            policy.model.train()
            loss = -policy.token_logps(inputs, completion).mean()
            scaler.scale(loss).backward()
            event["loss"] = loss.detach().item()
        else:
            completions, texts, rewards = [], [], []
            for _ in range(cfg["rollouts"]):
                tokens, text = policy.generate(inputs, sample=True)
                completions.append(tokens)
                texts.append(text)
                rewards.append(float(parse_r3(text) == target))
            total = sum(rewards)
            advantages = [r - (total-r) / (len(rewards)-1) for r in rewards]
            event.update(rewards=rewards, advantages=advantages, completions=texts,
                         completion_lengths=[t.numel() for t in completions])
            # Avoid spending the entire pilot optimizing KL-only groups at the base policy.
            if max(rewards) == min(rewards):
                event.update(skipped="zero_reward_variation", seconds=time.monotonic()-tick)
                append_jsonl(run / "train_log.jsonl", event)
                del inputs, completions
                continue
            policy.model.train()
            losses, kls = [], []
            for completion, advantage in zip(completions, advantages):
                with torch.no_grad(), policy.reference_context():
                    reference = policy.token_logps(inputs, completion)
                logps = policy.token_logps(inputs, completion)
                d = reference - logps
                kl = torch.exp(d) - d - 1
                loss = (-advantage * logps.sum() + cfg["kl_beta"] * kl.sum()) / (cfg["max_new_tokens"] * len(completions))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite policy loss")
                scaler.scale(loss).backward()
                losses.append(loss.detach().item())
                kls.append(kl.detach().mean().item())
                del logps, reference, kl, loss
            event.update(loss=sum(losses), sampled_kl=float(np.mean(kls)))
            del completions
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            event["skipped"] = "nonfinite_gradients"
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            event.update(step_after=step, grad_norm=float(norm))
        event.update(seconds=time.monotonic()-tick, **policy.memory())
        append_jsonl(run / "train_log.jsonl", event)
        print(json.dumps({k: v for k, v in event.items() if k != "completions"}), flush=True)
        del inputs
        if step > last_saved and step % cfg["save_every"] == 0:
            save_checkpoint(policy, optimizer, scaler, run, step, cursor)
            last_saved = step
    if step > last_saved:
        save_checkpoint(policy, optimizer, scaler, run, step, cursor)
    write_json(run / "status.json", {"optimizer_steps": step, "cursor": cursor, "elapsed_seconds": time.monotonic()-start,
                                     "complete_requested_steps": step >= args.steps, "usable_adapter": step > 0})

if __name__ == "__main__":
    main()
