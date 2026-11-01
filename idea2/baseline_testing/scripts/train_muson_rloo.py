"""Class-balanced RLOO continuation on MUSON egocentric action choices.

Starts from the R³-FDT video RLOO adapter. The six-way action label is the
only reward; annotation explanations are not used as a text target.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from social_rl.common import append_jsonl, config, read_jsonl, sha256, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import muson_prompt, parse_letter
from train import save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-adapter", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--max-hours", type=float, default=3.4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = config()
    rows = read_jsonl(args.manifest)
    if not rows or any(r["dataset"] != "muson" or r["split"] != "train" for r in rows):
        raise ValueError("Training requires MUSON train-only manifest")
    buckets = {c: [r for r in rows if r["answer"] == c] for c in range(6)}
    if any(not b for b in buckets.values()):
        raise ValueError("Every MUSON action class needs training examples")
    for c, bucket in buckets.items():
        random.Random(cfg["seed"] + c).shuffle(bucket)
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"config": cfg, "method": "rloo_muson_action_continuation",
                "manifest_sha256": sha256(args.manifest), "examples": len(rows),
                "blocks": len({r['group'] for r in rows}), "start_adapter": args.start_adapter,
                "sampling": "round-robin action classes; seeded shuffle within class",
                "seed": cfg["seed"]}
    if (run / "metadata.json").exists() and json.loads((run / "metadata.json").read_text()) != metadata:
        raise ValueError("Run metadata changed; refusing incompatible resume")
    if (run / "latest.json").exists() and not args.resume:
        raise ValueError("Existing checkpoint; explicitly pass --resume")
    write_json(run / "metadata.json", metadata)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    adapter = json.loads((run / "latest.json").read_text())["checkpoint"] if args.resume else args.start_adapter
    policy = VideoPolicy(adapter=adapter, trainable=True)
    params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    step, cursor = 0, 0
    if args.resume:
        state = torch.load(Path(adapter) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, cursor = state["step"], state["cursor"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.monotonic()
    last_saved = step
    attempts = 0
    while step < args.steps and time.monotonic() - started < args.max_hours * 3600:
        if attempts >= args.steps * 12:
            print("STOP: bounded attempts reached", flush=True)
            break
        attempts += 1
        action_class = cursor % 6
        row = buckets[action_class][(cursor // 6) % len(buckets[action_class])]
        cursor += 1
        permutation = random.sample(range(6), 6)
        target = permutation.index(row["answer"])
        inputs = policy.inputs(muson_prompt(permutation=permutation), image_path=row["image_path"])
        tick = time.monotonic()
        event = {"cursor": cursor, "step_before": step, "id": row["id"],
                 "block": row["group"], "action_class": action_class,
                 "permutation": permutation, "target": target}
        completions, texts, rewards = [], [], []
        for _ in range(cfg["rollouts"]):
            tokens, answer = policy.generate(inputs, sample=True)
            completions.append(tokens)
            texts.append(answer)
            rewards.append(float(parse_letter(answer, 6) == target))
        total = sum(rewards)
        advantages = [r - (total-r)/(len(rewards)-1) for r in rewards]
        event.update(rewards=rewards, advantages=advantages, completions=texts,
                     completion_lengths=[x.numel() for x in completions])
        if max(rewards) == min(rewards):
            event.update(skipped="zero_reward_variation", seconds=time.monotonic()-tick)
            append_jsonl(run / "train_log.jsonl", event)
            del inputs, completions
            continue
        optimizer.zero_grad(set_to_none=True)
        policy.model.train()
        losses, kls = [], []
        for completion, advantage in zip(completions, advantages):
            with torch.no_grad(), policy.reference_context():
                reference = policy.token_logps(inputs, completion)
            logps = policy.token_logps(inputs, completion)
            d = reference - logps
            kl = torch.exp(d) - d - 1
            loss = (-advantage*logps.sum() + cfg["kl_beta"]*kl.sum())/(cfg["max_new_tokens"]*len(completions))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite policy loss")
            scaler.scale(loss).backward()
            losses.append(loss.detach().item())
            kls.append(kl.detach().mean().item())
            del reference, logps, kl, loss
        event.update(loss=sum(losses), sampled_kl=float(np.mean(kls)))
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
        del inputs, completions
        if step > last_saved and step % cfg["save_every"] == 0:
            save_checkpoint(policy, optimizer, scaler, run, step, cursor)
            last_saved = step
    if step > last_saved:
        save_checkpoint(policy, optimizer, scaler, run, step, cursor)
    write_json(run / "status.json", {"optimizer_steps": step, "cursor": cursor,
               "elapsed_seconds": time.monotonic()-started,
               "complete_requested_steps": step >= args.steps, "usable_adapter": step > 0})


if __name__ == "__main__":
    main()
