# Gemma 4 12B speech-transfer experiment: how to run

Question: does steering built from TYPED remarks (an "untrustworthy" direction inside the model) also reduce appearance bias when the
same remark is SPOKEN? Plus an exploratory face arm (vision). Plan and success rule: `../QWEN_TRANSFER_PLAN.md` (Nov 25 update).
All code is in `../src/gemma_speech.py` (stages) and `../src/omni_model.py` (class `Gemma4`). The test split is locked.

Needs one GPU with at least 40 GB (A100 40/80 GB, H100, L40S 48 GB, RTX Pro 6000). An L4 (24 GB) is too small. About 4 to 6 hours.

## Option 1: Colab
1. Put `gemma_speech_bundle.tar.gz` in Google Drive at `My Drive/idea3_gemma_speech/`.
2. Colab: File > Upload notebook > `gemma_speech.ipynb`. Runtime > Change runtime type > A100.
3. Runtime > Run all. Allow Drive access (required: results are saved there).
4. If it disconnects or says "time limit reached": Run all again; it resumes.
5. Send back the `gemma_results_*.zip` it saves in that Drive folder, and the notebook with outputs.

## Option 2: any Linux GPU machine (AWS, lab server)
    cp gemma_colab/gemma_speech_bundle.tar.gz gemma_aws/      # the runner looks for the bundle next to itself
    cd gemma_aws && tmux new -s g && python3 aws_run.py       # resumable; results and zip in ~/idea3_gemma_speech/

Please do not change the code or settings without telling Yash: the selection rule and success rule were fixed before any results.
