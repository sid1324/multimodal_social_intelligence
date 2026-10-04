# Strict grader review of Analysis 1

## Verdict and scope

**The revised Analysis 1 contribution is defensible within its stated limits. The complete team submission is not certified.** The main matched comparison reproduces, and previously inaccessible source audits and frozen-feature analyses have now been performed. The strongest earlier causal wording was not defensible and has been removed. The package reports an unresolved old-probe reproduction discrepancy rather than hiding it.

No numerical grade is assigned: the supplied instructions specify expectations and page limits but do not provide a complete point rubric for this contribution. A strict review cannot infer the quality of the other teammates' analyses from these files.

## High-severity issues found and resolved

1. **Wrong choice space in the prior probe.** The old script dropped all empty-string candidates even though the official prompt makes none a valid choice and current gold selects it on 21 items. The primary probe now keeps all five choices. Chance is 20% for uniform five-choice action selection. Uniform nonempty-choice selection is 24.72% over the full set; 25% applies only conditional on nonempty gold. The old restricted results are marked historical.

2. **Mixed denominators in the grounding narrative.** The old main prose used the Pro/GPT pairwise gains beside a three-condition matched figure. Revised main gains are 27.1/16.2/19.0 points with the correct 1,851/1,156/1,194 populations. A separate 757-item intersection across all models confirms positive gains without asserting equal cross-model populations elsewhere.

3. **Potential evaluation-code confound.** `eval_api.py` formats the description prefix for the action prompt but uses the unformatted `self.prefix` for justification. It also treats index 4 as none even for a nonempty candidate. The current release has 34 examples whose correct nonempty action is at index 4. We cannot reconstruct historical prompts from output indices. The main section states this limitation, retains action-only sensitivity, and does not claim a clean causal demonstration of information loss.

4. **Unsupported untouched-benchmark claim.** A supervised diagnostic probe uses EgoNormia gold in cross-validation. The transfer model has not been trained on it, but the benchmark has been used for design analysis. Revised discussion explicitly makes that distinction and retains the untested mixture/order controls needed for the curriculum hypothesis.

5. **Unsupported comparisons across tasks.** Probe action accuracy is not compared as if it were VLM joint action/justification accuracy. Candidate-option norm prediction and scene-to-correct-option norm prediction are not treated as the same task. New visual/text AUROCs use identical items, folds, and correct-action targets.

6. **Overclaiming video and shortcuts.** Negative observed format deltas do not show that temporal evidence is unnecessary. The frame grid already supplies multiple times. A probe-error subset is not guaranteed to be free of all linguistic artifacts; a chance-level null does not prove there is no leakage. These claims are now qualified.

## Reproduction findings

The three raw EgoNormia SHA-256 hashes match the supplied archive. The original grounding ladder, per-category tables, paired comparisons, transitions, verified subsets, label counts, and gold sensitivity reproduce numerically, including the original bootstrap outputs when the original scripts are run.

Several old language-probe numbers do not reproduce. Supplied source/strict TF-IDF accuracies are 61.68%/60.60%; unchanged scripts here give 62.06%/61.58%. A separate environment with the old NumPy, pandas, and SciPy versions yields the same local rerun results. The old exact folds were not archived, and the cause remains unresolved. Changes in platform-dependent tied-group ordering are a possible explanation, not an established finding. The actual disagreement is preserved in `legacy_table_comparison.csv` and the original ZIP.

The revised five-choice probe is a separately specified analysis. Source/strict scores are 60.87%/59.96%. Its strict connected components are formed before fitting, using source identity plus exact normalized nonempty behavior text. Fold IDs are frozen; an additional rerun using those frozen IDs reproduces the primary probe and robustness CSVs byte-for-byte. The new failure set has 742 items, not 729; only 740/473/489 have all required model predictions. The corresponding joint gains are 16.1/17.1/15.5 points.

The corrected length baseline falls to 21.59%, illustrating why retaining none matters even when the substantive shortcut conclusion persists. MPNet action selection is 54.40%. Neither result is portrayed as a new zero-shot VLM experiment.

## Statistical scrutiny

- Primary CIs resample UUID-prefix clusters, not individual options. Cross-model pooled category intervals jointly resample shared source IDs. Category labels overlap; their counts are not summed as disjoint sample sizes.
- McNemar p-values are exact and use the same item populations as their deltas. Because item independence can fail within sources, source-level sign-swap sensitivity is included. The tests describe the released predictions; these are not randomized modality interventions.
- The two exploratory category contrasts use direct paired bootstrap differences, avoiding inference solely from interval overlap. They are not a preregistered multiple-comparison-corrected taxonomy ranking.
- Probe intervals resample fixed out-of-fold outcomes. They do not include model-refitting uncertainty. Three random-label seeds are a limited negative control, not an exhaustive null distribution.
- All fitted TF-IDF/SVD/scalers/classifiers stay within training folds. Frozen pretrained encoders are used only for feature extraction. Pretraining contamination cannot be ruled out from these files.
- The 400-item CLIP analysis has only 12 Privacy positives. No significance or general superiority claim is made from that category. Mean pooling removes temporal order; t-SNE is descriptive.
- A shared 400-item comparison yields macro AUROC 0.561 CLIP, 0.626 TF-IDF+LSA descriptions, and 0.552 MPNet descriptions. The overall option/description MPNet numbers answer different supervised questions and are labeled accordingly.

## New source-audit findings

**Hugging Face parity:** 1,743 shared IDs, 110 GitHub-only IDs, no Hugging-Face-only IDs, 27 description differences. The cause of missing rows is not claimed to be known.

**NormBank:** 155,423 rows; 32,059 exact duplicate rows; 1,166 context keys with conflicting labels. No exact setting/behavior/constraints key overlaps provided splits. This does not certify absence of semantic leakage. Parser-detected missingness is distinguished from semantic data quality.

**NormLens:** 934 high-agreement and 1,049 mid-agreement rows. There are 117 image-level tied majorities in the latter. High agreement means unanimity among available votes; four examples have only one retained vote. Repeated-action context statistics are descriptive and not called a text-only-to-image human judgment flip rate.

**VideoNorms:** the author-linked repository exposes only a README at the inspected commit; a public Hugging Face name search returns no datasets. The team reports that it has contacted the authors to request access and expects the annotations soon. This is a user-provided status update, not an independently verified delivery commitment. No VideoNorms annotation result is reported.

## Qualitative and artifact review

Two cases are selected by a fixed hash rule from predefined released-outcome strata, then inspected against actual downloaded pixels. The rescue case contains a fist bump absent from its generated description; it does not establish why the model changed its prediction. The failure case fixes the action but not the justification, so it is not called a perceptual failure or proof of missing temporal reasoning. The figure shows actual frames and preserves original source strips and full question records.

All current section sentences are indexed in `sentence_claim_ledger.csv`, with evidence or an explicit interpretation/future-work/status classification. `validation_results.json` records independent raw-count, paired-arithmetic, fold-integrity, representation-AUROC, and manuscript checks. Final PDF pages were rendered and visually inspected. The main preview and supplement compile from the included LaTeX and official ICML style files; final team float placement still requires checking after integration.

## Assignment readiness

The contribution contains hypotheses, quantitative analysis, plots, qualitative inspection, and an explicit discussion of what changes in Idea 1. The central result and the prompt caveat remain in the main section because the assignment says the appendix will not be graded. The main preview is a two-page standalone file including the human-exercise paragraph and references. The supplementary PDF has five pages including references, within the supplied optional appendix allowance before team additions. The introduction/discussion contribution and AI disclosure are separate editable sections.

The human exercise now includes one participant, 20 distinct-source items, and 60 responses. Gold-action agreement is 8/20, 8/20, and 9/20; mean confidence is 1.00, 4.00, and 4.05. Adding descriptions changes 16 choices (five gains, five losses, six changed disagreements); adding frames changes seven (three gains, two losses, two changed disagreements). Stage-to-stage exact agreement is 20% options/description, 65% description/frames, and 35% options/frames. These are within-person response consistencies, not inter-rater agreement. The first block scores 2/4/5 out of ten and the second 6/4/4. The extension was requested after first-block results and item-level feedback had been disclosed; this exposure is explicitly documented. No feedback was given within either block. All 60 choices and confidence values are checked against the conversation and all 20 gold labels against raw data. Disagreements are retained, including frame-induced losses at items 12 and 18. The evidence codebook, all reason-level tags, and overlapping counts are provided; codes are post-hoc single-AI-coder interpretations, not independently validated human labels. The sample now reaches 20 items, but remains a one-participant, fixed-order exploratory exercise, with no inter-person agreement or causal claim. The separate human-exercise PDF is supporting material, not an uncounted addition to the five-page assignment supplement.

For the five-person team, the overall report still needs four separate analyses, the team introduction and discussion, a valid repository link with the required access, truthful teammate contributions, and the required complete AI-use disclosure. TA consultation and those team facts are not verifiable from the supplied package. Do not claim this one-contribution ZIP establishes them. No external messages, submissions, or repository publication were performed.

## Residual scientific limits

The study does not test whether curriculum order improves transfer, does not provide fresh controlled prompts for historical VLM outputs, does not perform SFT, and does not establish that all shortcuts are removed. These are openly stated limits, not hidden incomplete results. Within that boundary, the package replaces the earlier overclaims with reproducible measurements and concrete changes to data preparation and evaluation.
