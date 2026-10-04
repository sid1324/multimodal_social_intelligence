"""Check new saved rows and notify the user of new recovery observations."""
import json,subprocess,time
from pathlib import Path
from check_egonormia_recoveries import main as check
from build_recovery_gallery import main as gallery

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/idea5-expanded'
while True:
    old=json.loads((OUT/'recovery-alerts.json').read_text())['alerts'] if (OUT/'recovery-alerts.json').exists() else []
    known={(x['workflow'],x['id'],x['combined_condition']) for x in old}
    try:
        check();gallery();new=json.loads((OUT/'recovery-alerts.json').read_text())['alerts']
    except (json.JSONDecodeError,FileNotFoundError):
        print('Snapshot incomplete; retrying after transient read failure.',flush=True)
        time.sleep(3)
        continue
    for row in new:
        if (row['workflow'],row['id'],row['combined_condition']) in known:continue
        extra=' with descriptions' if row['combined_condition']=='all_descriptions' else ' with maps only'
        message=f"{row['workflow']}: RGB wrong, all three maps{extra} correct. Case {row['id']}"
        print('USER RECOVERY ALERT: '+message,flush=True)
    results=json.loads((OUT/'qwen_results.json').read_text())
    if len(results)==640:break
    live=subprocess.run(['pgrep','-f','[c]ompare_expanded_qwen.py'],capture_output=True,text=True)
    if live.returncode!=0:print('Inference is no longer live; recovery monitor stopped. Resume only after inspecting state.',flush=True);break
    time.sleep(30)
