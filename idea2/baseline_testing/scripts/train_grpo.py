"""On-policy, two-epoch clipped GRPO pilot on R³-FDT video questions only.

Four completions share one question and a binary verified-answer reward.
Advantages are standardized within the group. Token ratios compare the policy
to a frozen snapshot taken before the group's updates; a fixed base policy
provides the sampled reference KL. This uses no critic or replay buffer.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from social_rl.common import STORE, append_jsonl, config, read_jsonl, sha256, write_json
from social_rl.grpo import clipped_token_objective, group_advantages
from social_rl.model import VideoPolicy
from social_rl.tasks import parse_r3, r3_prompt
from train import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=100,
                        help="Number of mixed-reward groups to optimize")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Policy updates over each sampled group")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-hours", type=float, default=3.4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.groups < 1 or args.epochs < 2 or not (0 < args.clip_epsilon < 1):
        parser.error("groups >= 1, epochs >= 2, and 0 < clip-epsilon < 1 are required")
    cfg = config()
    rows = read_jsonl(args.manifest)
    if not rows or any(r["dataset"] != "r3" or r["split"] != "train" for r in rows):
        raise ValueError("Training requires the R³-FDT train-only manifest")
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    metadata = {"config": cfg, "method": "grpo", "manifest_sha256": sha256(args.manifest),
                "examples": len(rows), "videos": len({r["video_id"] for r in rows}),
                "seed": cfg["seed"], "epochs": args.epochs, "clip_epsilon": args.clip_epsilon}
    if (run / "metadata.json").exists() and json.loads((run / "metadata.json").read_text()) != metadata:
        raise ValueError("Run metadata changed; refusing incompatible resume")
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
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    groups, cursor = 0, 0
    if adapter:
        state = torch.load(Path(adapter) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        groups, cursor = state["step"], state["cursor"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    prior_events = read_jsonl(run / "train_log.jsonl") if (run / "train_log.jsonl").exists() else []
    optimizer_updates = sum(e.get("accepted_epochs", 0) for e in prior_events)
    started = time.monotonic()
    last_saved = groups
    attempted = 0
    while groups < args.groups and time.monotonic() - started < args.max_hours * 3600:
        if attempted >= args.groups * 10:
            print("STOP: bounded attempts reached without enough mixed-reward groups", flush=True)
            break
        row = rows[order[cursor % len(order)]]
        cursor += 1
        attempted += 1
        permutation = random.sample(range(5), 5)
        target = permutation.index(row["answer"])
        inputs = policy.inputs(r3_prompt(row, permutation), row["video_path"],
                               end_seconds=row.get("clip_duration_seconds"))
        tick = time.monotonic()
        event = {"cursor": cursor, "group_before": groups, "id": row["id"],
                 "video_id": row["video_id"], "permutation": permutation,
                 "target": target, "method": "grpo"}
        completions, texts, rewards = [], [], []
        for _ in range(cfg["rollouts"]):
            tokens, answer_text = policy.generate(inputs, sample=True)
            completions.append(tokens)
            texts.append(answer_text)
            rewards.append(float(parse_r3(answer_text) == target))
        advantages = group_advantages(rewards).tolist()
        event.update(rewards=rewards, advantages=advantages, completions=texts,
                     completion_lengths=[t.numel() for t in completions])
        if max(rewards) == min(rewards):
            event.update(skipped="zero_reward_variation", seconds=time.monotonic() - tick)
            append_jsonl(run / "train_log.jsonl", event)
            del inputs, completions
            continue

        # Snapshot the behavior policy once, before either optimization epoch.
        policy.model.eval()
        old_logps, ref_logps = [], []
        for completion in completions:
            with torch.no_grad():
                old_logps.append(policy.token_logps(inputs, completion).detach())
                with policy.reference_context():
                    ref_logps.append(policy.token_logps(inputs, completion).detach())
        epoch_losses, epoch_kls, epoch_clipped = [], [], []
        accepted = 0
        for epoch in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            policy.model.train()
            losses, kls, clipped = [], [], []
            for completion, old, ref, advantage in zip(completions, old_logps, ref_logps, advantages):
                logps = policy.token_logps(inputs, completion)
                loss, kl, fraction = clipped_token_objective(
                    logps, old, ref, advantage, args.clip_epsilon, cfg["kl_beta"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite GRPO objective")
                scaler.scale(loss / len(completions)).backward()
                losses.append(loss.detach().item())
                kls.append(kl.item())
                clipped.append(fraction.item())
                del logps, loss
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                scaler.step(optimizer)
                scaler.update()
                event.setdefault("nonfinite_epochs", []).append(epoch)
            else:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                accepted += 1
                optimizer_updates += 1
                event.setdefault("grad_norms", []).append(float(norm))
            epoch_losses.append(float(np.mean(losses)))
            epoch_kls.append(float(np.mean(kls)))
            epoch_clipped.append(float(np.mean(clipped)))
        event.update(accepted_epochs=accepted, epoch_losses=epoch_losses,
                     epoch_sampled_kl=epoch_kls, epoch_clipped_fraction=epoch_clipped,
                     optimizer_updates=optimizer_updates)
        if accepted:
            groups += 1
            event["group_after"] = groups
        else:
            event["skipped"] = "all_epochs_nonfinite"
        event.update(seconds=time.monotonic() - tick, **policy.memory())
        append_jsonl(run / "train_log.jsonl", event)
        print(json.dumps({k: v for k, v in event.items() if k != "completions"}), flush=True)
        del inputs, completions, old_logps, ref_logps
        if groups > last_saved and groups % cfg["save_every"] == 0:
            save_checkpoint(policy, optimizer, scaler, run, groups, cursor)
            last_saved = groups
    if groups > last_saved:
        save_checkpoint(policy, optimizer, scaler, run, groups, cursor)
    write_json(run / "status.json", {
        "optimized_groups": groups, "optimizer_updates": optimizer_updates, "cursor": cursor,
        "elapsed_seconds": time.monotonic() - started,
        "complete_requested_groups": groups >= args.groups, "usable_adapter": groups > 0,
    })


if __name__ == "__main__":
    main()
