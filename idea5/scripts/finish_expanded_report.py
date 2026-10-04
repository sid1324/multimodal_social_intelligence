"""Build dependent deliverables after the resumable inference process completes."""
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'

def main():
    while True:
        path=OUT/'qwen_results.json'
        if path.exists() and len(json.loads(path.read_text()))==640:break
        process=subprocess.run(['pgrep','-f','[c]ompare_expanded_qwen.py'],capture_output=True,text=True)
        if process.returncode!=0:raise RuntimeError('Inference process is not live and fewer than 640 rows are saved; inspect qwen.log before resuming.')
        time.sleep(30)
    for cmd,cwd in [([str(ROOT/'.venv/bin/python'),'scripts/build_expanded_report.py'],ROOT),([str(ROOT/'.venv/bin/python'),'scripts/verify_expanded_experiment.py'],ROOT),(['tectonic','analysis-preview.tex'],OUT)]:
        subprocess.run(cmd,cwd=cwd,check=True)
    render=OUT/'rendered';render.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-r','110','-png',str(OUT/'analysis-preview.pdf'),str(render/'page')],check=True)
    print('Report built and rendered; root visual inspection remains required.',flush=True)

if __name__=='__main__':main()
