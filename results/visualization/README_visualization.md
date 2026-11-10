# DLBA — visualization-ready results pack

**Source:** The uploaded `results(1).zip` only. No new inference, retraining, or statistical resampling was run to create these tables. Data join and aggregation checks run in `build_dlba_visualization_data.py`.

## Where to start

**`DLBA_visualization_MASTER.csv`** — 90 rows, **one row per (seed, heuristic, HANS gold label, subcase)**. This is the single file to import into a flexible visual tool (Tableau, Flourish, Power BI, Plotly). It includes each subcase's accuracy, flip direction and probability changes, plus its parent heuristic-label metrics and contextual per-seed MNLI accuracy. The GPU runtime values are populated **only for seed 42** and are labeled as CUDA eager.

Other tables give cleaner sources for particular panels:

| File | Rows | Grain and purpose |
| --- | ---: | --- |
| `DLBA_heuristic_label_18rows.csv` | 18 | 3 seeds × 3 heuristic families × 2 gold labels. Accuracy, paired bootstrap 95% CI, McNemar exact p, BH q, harmful/beneficial flips, probability change, 3-seed summary statistics. Best for the main facet/heatmap/forest plot. |
| `DLBA_subcase_drilldown_90rows.csv` | 90 | 3 seeds × 30 HANS subcases. Detail heatmap, nested slope charts, drilldown and confidence changes. |
| `DLBA_crossseed_heuristic_6rows.csv` | 6 | Across-seed group means and min/max, consistency, total flips. Best for headline comparison. |
| `DLBA_crossseed_subcase_30rows.csv` | 30 | Across-seed subcase means and consistency counts. Best for highlighting the most sensitive templates. |
| `DLBA_seed_overview_3rows.csv` | 3 | MNLI matched/mismatched vs HANS overall by seed; CUDA seed-42 runtime as available. Best for top-level KPI cards/slope charts. |
| `DLBA_runtime_cuda_seed42_2rows.csv` | 2 | Direct measured FP32 eager vs INT8 eager throughput, memory, latency and storage on seed-42 CUDA; not a multi-seed estimate. |
| `DLBA_entailment_probability_shift_bins.csv` | 324 | Binned per-example INT8-minus-FP32 probability-of-entailment changes. Each bin includes sample share, density and harmful/beneficial flips. |
| `DLBA_illustrative_flip_examples.csv` | Varies | Up to three illustrative high-magnitude flip examples for each seed × heuristic × label × flip type; these are selected examples, **not a representative random sample**. |

## A sophisticated yet defensible visualization

Create a multi-panel **quantization behavioral impact dashboard**:

1. **Top strip — baseline validity.** FP32 vs INT8 MNLI matched, mismatched and HANS overall by seed using `DLBA_seed_overview_3rows.csv`. Label changes in **percentage points**.
2. **Centerpiece — 3 × 2 HANS behavioral matrix.** x=heuristic, y=gold label; color=`delta_accuracy_mean_pp` using `DLBA_crossseed_heuristic_6rows.csv`. Make marks expandable to per-seed dot-and-CI forest plots using `DLBA_heuristic_label_18rows.csv`. Marker size=HANS flip rate, separate labels for harmful vs beneficial flips. This reveals how average accuracy can mask subgroup degradation.
3. **Below — subcase variability.** Use `DLBA_crossseed_subcase_30rows.csv`: ordered diverging bars of mean accuracy change, min–max whiskers, tooltip for `seeds_with_drop`. Drill down to 90 per-seed rows.
4. **Right panel — confidence shifts.** For a selected subgroup, use `DLBA_entailment_probability_shift_bins.csv` to show the full distribution of INT8−FP32 predicted entailment probability. Because bins have unequal widths, use `density_per_pp` for a continuous histogram or the shares for categorical interval bars.
5. **Footer — efficiency cost.** Compare the seed-42-only CUDA eager model storage and peak memory savings against its throughput slowdown. **Do not** graph these as CPU quantization speed measurements or across-seed averages.
6. **Qualitative tooltip.** Display sentence-pair examples from `DLBA_illustrative_flip_examples.csv` for relatable illustrations, distinguishing correct→wrong from wrong→correct.

## Metric glossary

- `fp32_accuracy_pct`, `int8_accuracy_pct`: share of correct examples ×100.
- `delta_accuracy_pp`: `INT8 accuracy − FP32 accuracy`, in percentage points; negative means INT8 is worse.
- `mean_delta_prob_entailment_pp`: average `100 × (P_INT8(entailment) − P_FP32(entailment))`; probability shifts do not by themselves prove shortcut learning.
- `mean_abs_delta_prob_entailment_pp`: average absolute probability change ×100 (NOT absolute of the average).
- `correct_to_wrong` and `wrong_to_correct`: paired correctness transitions at the HANS binary label level. `net_harm_count` = harmful minus beneficial flips.
- `mcnemar_p_raw`: exact paired McNemar test p for binary correctness changes. `mcnemar_bh_q_18_subgroup_tests` is Benjamini–Hochberg adjusted over the 18 seed×heuristic×label tests **only**, newly calculated from the supplied p-values. Neither p nor q quantifies effect size.
- `accuracy_ci95_low_pp`/`high_pp`: supplied per-subgroup paired bootstrap 95% CI endpoints, NOT adjusted for 18 comparisons. Across-seed SD describes only 3 training seeds.
- `*_pred_entailment_pct`: fraction **predicted entailment**, not probability. On non-entailment examples this equals the error rate; the same interpretation does not apply on entailment examples.
- `hans` HANS labels have 2 classes but BERT predicts 3 MNLI classes: neutral and contradiction map to non-entailment.

## Interpretation boundaries

- FP32 main HANS predictions were saved from GPU inference and adopted dynamic INT8 HANS predictions from CPU inference. CPU-FP32 control was intentionally not performed. **Do not claim** numerical device equivalence was verified.
- The published GPU eager runtime benchmark is **seed 42 only** and quantized GPU kernels may be slower; it is not the same execution setting as the main CPU INT8 HANS accuracy evaluation.
- Repeating parent-group and seed-level metrics inside the 90-row master simplifies dashboard construction but causes double counting when summed. For group or seed KPIs, use the dedicated 18-, 6-, or 3-row tables, or use FIRST per unique key (not SUM).
- McNemar significance of many individual examples cannot by itself establish that effects generalize across training seeds; inspect signs and magnitudes across the three seeds.
- HANS accuracy shifts are evidence of behavior correlated with heuristics, not direct identification of the model's internal features.
- The `DLBA_entailment_probability_shift_bins.csv` uses **unequal bin widths** intentionally to resolve the large cluster near zero. When plotting as a continuous histogram, use density rather than raw count per bin width.
- All percentage fields are numeric 0–100; **do not use percentage number formatting that multiplies values by 100 again**.

## Reproduce

Place `results(1).zip` under `/mnt/data/results(1).zip` and run:

```bash
python build_dlba_visualization_data.py
```

The script reads only selected CSV/JSON members (not the large profiler Chrome traces), verifies 90,000 matched HANS predictions by `pairID`, and asserts count consistency with the subgroup and subcase summaries.
