# 05. What the Midterm Report Needs from Idea 3

Course rubric summary (from the instructions PDF; read it for full detail): the midterm report follows a research-paper structure. Each of the N-1 = 4 research ideas needs (1) an **initial implementation**, meaning a baseline that is a first implementation of the idea using recent SOTA approaches, with math aligned to the problem statement; (2) **results and discussion**, with a table of metrics **on a test set different from training**, where the **analysis of failure cases is central** (ablations, qualitative analysis of failures related to multimodal concepts, hypotheses for why they fail and how to improve); and (3) **updated research ideas** for the final, which must apply course multimodal concepts and not be mere config variations, ideally with math. **Only the main pages are graded** (11 pages for the 5-person team, shared across all ideas). The appendix (up to 5 pages) is not graded. A separate page discloses all AI use plus teammate contributions.

## Idea 3's share (about 2.5 pages) and what feeds it
| Report part | Length | Needs from Claude Code |
|---|---|---|
| Related work: a category paragraph on appearance-based bias in VLMs, face impressions, debiasing via representation steering | ~1 paragraph, 4 to 5 citations | Nothing (written in chat) |
| Problem statement contribution (notation from `01_PROJECT_BRIEF.md`) | a few lines | Nothing |
| Methodology for Idea 3: data subset, splits, metrics | ~1/4 page | `splits.json` counts, usable-face count, spot-check agreement |
| **Initial implementation: directional ablation (naive and refined), constitution baseline** | ~1 page | the frozen configs, the direction cosine figure (F1), the sweep trade-off figure (F2) |
| **Results and discussion** | ~0.75 to 1 page | the main table, F3, F4, per-norm results, and **`failure_cases.md`** with qualitative examples |
| Updated idea: DPO vs. steering, conditional vs. unconditional, cross-modal transfer | ~0.5 page | the findings that motivate it (e.g. the cosine result, behavior-control damage) |

## Figure and table budget
Main pages are tight, so plan for **1 table + 2 figures** in the main report: the main results table, F2 (the trade-off) or F1 (the cosine), and F4 (face-text alignment) or one qualitative example. Make figures clean, readable at single-column ICML width (3.25 in), colorblind-safe, with axis labels in words. Export each figure as PDF and PNG.

## The AI disclosure log
The course asks for a detailed description of ALL AI use (code written, experiments run, text drafted) and stresses that students lead. Keep `AI_USAGE_LOG.md` current: what Claude Code wrote or ran, which decisions Yash made, and which gateway models were used as experimental subjects (their outputs are data, not authored text). Yash's own contributions (design decisions, human spot-check labels, review) are recorded separately by him.
