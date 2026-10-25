#!/bin/zsh
# Re-run until every condition has every item; a run killed by the OS (no message) is retried like a budget exit.
cd "$(dirname $0)/.."
for n in {1..60}; do
  .venv-embed/bin/python scripts/midterm_embed.py --conds X3,X2,C3,C4,C6 --split test,train,train_ext --max-forwards 300 >> output/idea5-midterm/embed_runs.log 2>&1
  missing=0
  for c in X3 X2 C3 C4 C6; do [ $(ls output/idea5-midterm/emb/$c | wc -l) -ge 798 ] || missing=1; done
  [ $missing = 0 ] && break
done
echo RESUME_DONE; date
