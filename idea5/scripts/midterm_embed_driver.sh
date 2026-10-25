#!/bin/zsh
# Re-embed everything with the verified MPS settings, restarting the process on its forward budget or on OOM.
cd "$(dirname $0)/.."
run() { while .venv-embed/bin/python scripts/midterm_embed.py "$@" --max-forwards 300 | tee -a output/idea5-midterm/embed_runs.log | grep -q FORWARD_BUDGET_REACHED; do :; done; echo "PASS_DONE $*"; }
.venv-embed/bin/python scripts/midterm_embed.py --options --split test,train,train_ext
echo OPTIONS_DONE
run --conds C0,C1,C2,C5,X1,C3,C4,X3,C6,X2 --split test
run --conds C0,C1,C2,C5,X1,C3,C4,X3,C6,X2 --split train,train_ext
echo ALL_EMBEDDINGS_DONE
