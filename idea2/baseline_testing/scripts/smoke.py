"""Synthetic inputs verify infrastructure only; never reported as research results."""
import json
import time
import torch
from social_rl.common import STORE, read_jsonl, write_json
from social_rl.model import VideoPolicy
from social_rl.tasks import r3_prompt

t = time.monotonic()
torch.manual_seed(777)
report = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
          "torch": torch.__version__, "engineering_smoke_only": True}
policy = VideoPolicy(trainable=True)
report["load_seconds"] = time.monotonic() - t
print("MODEL_LOADED", report, flush=True)
inputs = policy.inputs("Describe the changing color in this short video in one sentence.", synthetic=True)
report["input_tokens"] = inputs["input_ids"].shape[1]
tokens, text = policy.generate(inputs, max_new_tokens=24)
print("GENERATION", repr(text), flush=True)
report["generated_tokens"] = tokens.numel()
optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=1e-5)
policy.model.train()
logps = policy.token_logps(inputs, tokens)
loss = -logps.mean()
if not torch.isfinite(loss):
    raise RuntimeError("Non-finite smoke loss")
loss.backward()
grads = [p.grad for p in policy.model.parameters() if p.grad is not None]
if not grads or not all(torch.isfinite(g).all() for g in grads):
    raise RuntimeError("Missing or non-finite adapter gradients")
report["nonzero_grad_tensors"] = sum(int(torch.count_nonzero(g) > 0) for g in grads)
if report["nonzero_grad_tensors"] == 0:
    raise RuntimeError("All adapter gradients zero")
torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
rows = read_jsonl(STORE / 'data/manifests/r3_train_accessible.jsonl')
if not rows:
    raise RuntimeError('Real R3 training video is required for smoke test')
row = rows[0]
real = policy.inputs(r3_prompt(row), row['video_path'], end_seconds=row.get('clip_duration_seconds'))
real_tokens, real_text = policy.generate(real, max_new_tokens=16)
policy.model.train()
real_loss = -policy.token_logps(real, real_tokens).mean()
if not torch.isfinite(real_loss):
    raise RuntimeError('Non-finite real-video smoke loss')
real_loss.backward()
real_grads = [p.grad for p in policy.model.parameters() if p.grad is not None]
if not real_grads or not all(torch.isfinite(g).all() for g in real_grads):
    raise RuntimeError('Missing or non-finite real-video adapter gradients')
report['real_video'] = {'id': row['video_id'], 'input_tokens': real['input_ids'].shape[1],
                        'generated_tokens': real_tokens.numel(), 'loss': real_loss.item()}
optimizer.zero_grad(set_to_none=True)
del real, real_tokens
with torch.no_grad(), policy.reference_context():
    ref = policy.token_logps(inputs, tokens)
assert torch.isfinite(ref).all()
report.update(policy.memory())
report["loss"] = loss.item()
report["total_seconds"] = time.monotonic() - t
out = STORE / "runs/smoke"
policy.save(out / "adapter")
policy.model.eval()
with torch.no_grad():
    expected = policy.token_logps(inputs, tokens).detach().clone()
policy.model.load_adapter(out / "adapter", adapter_name="reload_test", is_trainable=False)
policy.model.set_adapter("reload_test")
with torch.no_grad():
    restored = policy.token_logps(inputs, tokens)
if not torch.allclose(expected, restored, atol=1e-4, rtol=1e-4):
    raise RuntimeError("Adapter reload changed token probabilities")
report["adapter_reload_verified"] = True
report.update(policy.memory())
report["total_seconds"] = time.monotonic() - t
write_json(out / "result.json", report)
print("SMOKE_PASS", json.dumps(report), flush=True)
