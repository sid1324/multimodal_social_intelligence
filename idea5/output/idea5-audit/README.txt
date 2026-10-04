EgoNormia / Idea 5 computational deliverable

Start with review.html for the charts, original frames, overlays, per-item model
results, and a form for your own inspection notes. Notes are stored only in the
browser and can be exported as JSON. Human review has not been performed.

analysis-preview.pdf is a standalone draft, not the complete ICML team report.
idea5-section.tex is the section to insert into the team's existing Overleaf file.
Upload coverage.png and idea5-references.bib alongside it. Merge the BibTeX entry
into the existing bibliography and reuse its existing citation key if needed.
qwen_comparison.png is an additional figure available for the report/appendix.
Replace the provisional discussion after personally reviewing the examples.
The preview includes an honest AI usage record to adapt to the team disclosure.

Measured scope
- 40 examples from 40 distinct source videos, fixed by hash before inference.
- Five pre-action frames per clip, 200 frames total.
- Frozen MediaPipe body and hand landmark extraction; 600 exported JPEGs.
- Frozen, local 4-bit Qwen3-VL-8B-Instruct; no fine-tuning.
- Four input conditions: RGB, repeated RGB, aligned pose, shuffled pose.
- Two sequential tasks per condition (action then justification): 320 responses.
- Independent fixed action/justification option permutations.
- Full prompt/response records and original-index scoring saved.
- No descriptions, correct labels, or taxonomy supplied as model inputs.
- The sample is exploratory; no full-benchmark or pose-accuracy claim.

Reproduce from /Users/patrickjain/Desktop/CMU/sem3/multimodal
  .venv/bin/python scripts/download_egonormia.py
  .venv-pose/bin/python scripts/audit_egonormia_pose.py
  .venv/bin/python scripts/compare_qwen_pose.py
  .venv-pose/bin/python scripts/build_audit_report.py
  cd output/idea5-audit
  tectonic analysis-preview.tex --keep-logs

The Qwen script resumes completed conditions from qwen_results.json. To perform
a fresh run, archive that file under another name before running the script.
The saved source revisions and installed package lists make the run traceable.
pose-requirements.txt and vlm-requirements.txt describe two isolated environments.
MediaPipe 1.0.1 failed during native macOS initialization; the successful pose
run used MediaPipe 0.10.21. The Qwen environment is separate from the pose run.

Dataset
  data/EgoNormia: full pinned Hugging Face release, 7,417 files, 65,229,810,036 bytes.
  data/EgoNormia/download_verification.json: every release file checked by size.
  data/EgoNormia/annotations/final_data.json: official 1,853-item benchmark labels.
  data/EgoNormia/annotations/verified_split.json: official verified subset IDs.
  data/EgoNormia/annotations/provenance.json: pinned evaluation-code revision.

Results and checks
  protocol.json: sample, clip selection, detector setup, thresholds.
  frame_results.json: timestamps, raw landmarks, confidences, exported paths.
  runtime.json: runtime, detector model hashes, platform versions.
  qwen_protocol.json: frozen model revision, prompting/decoding/control setup.
  qwen_results.json: all predictions, prompts, permutations, timings and scores.
  summary.json: counts, source bootstrap intervals, exploratory paired tests.
  verification.json: checked scoring, mapping, dimensions and prompt-token match.

Human tasks before submission
  Confirm the analysis scope with the primary TA.
  Review overlays and task relevance; automated coverage is not correctness.
  Lead and revise the interpretation, acknowledge limitations and AI assistance.
  Integrate the section into the team's shared page limit and contribution record.
