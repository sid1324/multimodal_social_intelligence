# 04. PSC Bridges-2 Setup (revised with lessons from Yash's 11-868 coursework on Bridges-2)

## Confirmed facts (Sat Oct 24, 2026)
- 11-777 allocation: **`cis260338p`** ("Grounded Social Norm Reasoning in Vision-Language Models", PI Atharv Patawar). At the start: 1,969 / 2,000 GPU SU left and 2,581 / 3,000 GB left. Yash (ylothe) is an active user.
- `P777=/ocean/projects/cis260338p/ylothe` is already in Yash's `~/.bashrc`. The team repo is cloned at `$P777/multimodal_social_intelligence`.
- The team-shared folder `/ocean/projects/cis260338p/shared/` is group-writable. **Model weights go in `/ocean/projects/cis260338p/shared/hf_cache`** (`HF_HOME`) so teammates can reuse them. No Qwen weights were present when checked.
- The login node's network was fast at about 10:57 am (git clone at 5.5 MB/s), so the 17 GB model should take about an hour.
- Bridges-2 shuts down **Nov 9, 2026** for 4 to 6 weeks (migration). Backups are due by Nov 1. Expect longer queues before then. This doesn't affect this weekend beyond congestion.

## Lessons from Yash's earlier PSC use (read first)
1. **Yash has more than one PSC allocation.** His 11-868 course uses `cis260267p` (`/ocean/projects/cis260267p/ylothe/`). **Do NOT use it for this project.** The 11-777 team allocation has a different charge ID. Get it from `projects`.
2. **`$PROJECT` may point to the 11-868 allocation** (it points to the default one). **Never use `$PROJECT` in this project.** Use the explicit variable `P777=/ocean/projects/cis260338p/ylothe`, set in `~/.bashrc` and in every sbatch script. **Always pass `-A cis260338p`** to `interact` and `sbatch`.
3. **The 11-777 credits must already be exchanged for GPU hours and storage** (ACCESS dashboard, "Exchange credits for resources"). Atharv proposed the split; confirm it was actually done, or jobs and storage under that allocation will fail.
4. **Use `uv` venvs, not conda envs,** with caches in project storage. Conda in home hit disk-quota problems before. The proven sequence: `module load cuda/12.4.0`, set `PIP_CACHE_DIR`/`UV_CACHE_DIR` into project storage, `uv venv --python 3.11 --seed .venv`.
5. **PSC login-node internet is sometimes very slow** (77 kB/s observed). The model download is about 17 GB, so **run downloads detached with `nohup`** and check the speed early. If it's crawling, use the Colab fallback (below).
6. **The node `v005` (GPU-shared, v100-32) had a broken NVML driver** during 11-868. Use `--exclude=v005` with v100-32. The class switched to `v100-16`, but **Qwen3-VL-8B in fp16 does not fit in 16 GB** (it would need 8-bit loading, which slows things and complicates hooks). Prefer v100-32 with `--exclude=v005`.
7. **GPU-shared gets congested near deadlines** (start times of a day or more were seen during 11-868 deadline nights). Submit early, keep jobs short (2 to 4 h), and shard work so partial results are usable. `GPU-dev` was restricted.
8. **Interactive sessions are capped at 8 hours; sbatch jobs can run up to 48 hours.**
9. **Laptop to PSC file transfer:** `scp` from PSC's side failed before (`scp: command not found` with `-O`). The reliable path is **VS Code Remote-SSH** (drag files into the explorer to upload; right-click and choose Download to download).
10. **Cancel stray jobs** with `scancel -u ylothe`.
11. `bridges2.psc.edu` can put you on a different login node each time. `ps` and `tmux` sessions are per node, but files in `/ocean` are visible from all of them. For long detached work, prefer `nohup` plus log files, and note the login node name (`hostname`) for tmux.

## Fallback: Google Colab A100
Yash has about 80 Colab compute units left. If PSC queues or network block progress for more than about 1 to 2 hours, run Steps 0 to 4 on a Colab A100 (bf16, much faster than a V100). Keep the code device-agnostic (no hard-coded PSC paths: read `DATA_DIR`, `CACHE_DIR` and `OUT_DIR` from environment variables) so switching costs minutes, not hours. Note the hardware used in the report.

## Commands (each with what it does)
```bash
projects
# Shows all your allocations with their charge IDs and storage paths. Find the 11-777 one (NOT cis260267p).

echo 'export P777=/ocean/projects/cis260338p/ylothe' >> ~/.bashrc && source ~/.bashrc
# Saves a shortcut to your 11-777 project folder so nothing lands in the 11-868 allocation by mistake.

mkdir -p $P777 && cd $P777
# Creates and enters your 11-777 project folder.

module load cuda/12.4.0
# Loads the CUDA version that worked in your 11-868 setups.

export PIP_CACHE_DIR=$P777/.cache/pip UV_CACHE_DIR=$P777/.cache/uv HF_HOME=/ocean/projects/cis260338p/shared/hf_cache
# Keeps pip, uv and HuggingFace downloads in project storage instead of your small home quota (add to ~/.bashrc too).

uv venv --python 3.11 --seed .venv && source .venv/bin/activate
# Creates and turns on a fast uv Python environment in the project folder.

nohup uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 > torch_install.log 2>&1 &
# Installs PyTorch built for CUDA 12.4 in the background so it survives disconnects; check progress with `tail torch_install.log`.

uv pip install "transformers>=4.57" accelerate qwen-vl-utils pillow numpy pandas scipy matplotlib pyyaml openai huggingface_hub tqdm opencv-python-headless
# Installs the remaining libraries after torch finishes (verify the minimum transformers version on the Qwen3-VL model card).

nohup huggingface-cli download Qwen/Qwen3-VL-8B-Instruct > model_download.log 2>&1 &
# Downloads the ~17 GB model weights in the background; `tail model_download.log` shows progress and speed.

sinfo -p GPU-shared -o "%N %G %t"
# Lists GPU-shared nodes and their GPU types, to confirm v100-32 availability and check v005's state.

interact -p GPU-shared --gres=gpu:v100-32:1 -t 02:00:00 -A cis260338p --exclude=v005
# Gets an interactive 2-hour session on one 32 GB V100 for the smoke test, charged to 11-777 and avoiding the broken node (if interact rejects --exclude, use an sbatch job with #SBATCH --exclude=v005).

nvidia-smi
# Inside the GPU session: confirms the GPU type, memory and that the driver works.
```

## Batch jobs
Each `slurm/*.sbatch` needs: `#SBATCH -p GPU-shared`, `#SBATCH --gres=gpu:v100-32:1`, `#SBATCH --exclude=v005`, `#SBATCH -A cis260338p`, `#SBATCH -t 03:00:00` (keep short), and `#SBATCH -o logs/%x_%j.out`. The body runs `module load cuda/12.4.0`, activates `$P777/.venv` and exports `HF_HOME`. Shard by method or item chunk so jobs run in parallel and partial results are usable.

```bash
sbatch slurm/eval_text.sbatch
# Submits a batch job.

squeue -u ylothe
# Shows your jobs and whether they are pending or running.

squeue -u ylothe --start
# Shows the estimated start time of pending jobs (use it to spot congestion early).

tail -f logs/<jobname>_<jobid>.out
# Watches a job's output live.

scancel -u ylothe
# Cancels all your jobs (use carefully).
```

## Fitting the model
- **V100-32GB:** fp16 fits (about 17 GB of weights). **No bf16 on V100.** Check for NaN/inf in the smoke test.
- **V100-16GB:** fp16 does not fit, so avoid it.
- **Colab A100:** bf16.

## Gateway (frontier models)
`.env` (`LITELLM_BASE_URL`, `LITELLM_API_KEY`) is uploaded by Yash through VS Code Remote-SSH, placed at `idea3_face_bias/midterm_steering/.env`, set to `chmod 600`, never committed or printed. Gateway calls run from the login node (light I/O).
