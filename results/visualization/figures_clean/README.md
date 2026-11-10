# DLBA visualizations

All eight figures use the existing visualization CSVs and are saved as PNG and SVG.

- Figures 01-07 use means over seeds 42, 123, and 456.
- The min-max range is the observed spread of three seeds, not a confidence interval.
- Figure 04 displays mean flips per 5000-example subgroup *per seed*; Figure 06 shows mean flips per seed and the mean example share across seeds.
- Figure 08 uses seed-42 CUDA eager benchmarking only (not CPU, not a seed average).
- All pp changes mean INT8 minus FP32.
- Figure 06 shows variable-width categorical probability intervals, not an equal-width density histogram.
- Main HANS FP32 and INT8 predictions were made on GPU and CPU, respectively; device equivalence was not independently verified.
- Group-level effects suggest heuristic sensitivity, not proof of internal shortcut features.
