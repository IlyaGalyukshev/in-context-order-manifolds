# BCS benchmark strata

*Generated from `data/bcs` — analysis-ready breakdown of every query category. Interior-answerable = all involved entities in ranks 3..N−2 (the confound-clean subset used for geometry-linked behavior). Endpoint-diagnostic = leaks role/frequency, reported as a control only.*

- families: `s0_zib, s1_size, s1_loud`  ·  N-grid: `[7, 9, 12, 16]`  ·  per-cell: 120  ·  difficulty: both  ·  degree: 4
- gate_failures at generation: **0**

- **stimuli:** 8640  {'total_order': 5760, 'partial_order': 1440, 'grid2d': 1440}
- **coherence-null twins:** 2880
- **questions:** 267840  (orphans: 0)
- **degeneracy flags:** none

## Per-category strata

| family | n | interior | interior % | endpoint | #keys | top-key share | chance |
|---|--:|--:|--:|--:|--:|--:|---|
| `betweenness` | 17280 | 7200 | 41.7 | 0 | 472 | 0.004 | 0.33 (3-choice) |
| `comparative_distance` | 17280 | 7920 | 45.8 | 0 | 472 | 0.004 | 0.50 (2-choice) |
| `count_between` | 17280 | 8640 | 50.0 | 0 | 15 | 0.256 | const-0 ~26%; report by rank_distance / MAE, not 1/(N-1) |
| `extremes` | 5760 | 0 | 0.0 | 5760 | 469 | 0.005 | ~1/N (endpoint-diagnostic) |
| `order_query` | 34560 | 0 | 0.0 | 0 | 473 | 0.5 | 0.50 (2-choice + 'undetermined') |
| `pairwise` | 116640 | 31680 | 27.2 | 0 | 472 | 0.004 | 0.50 (2-choice) |
| `predecessor` | 14400 | 7920 | 55.0 | 0 | 472 | 0.004 | ~1/N |
| `rank` | 24480 | 15254 | 62.3 | 4613 | 488 | 0.049 | ~1/N |
| `reconstruction` | 5760 | 0 | 0.0 | 0 | 1441 | 0.5 | Kendall tau = 0 |
| `successor` | 14400 | 7920 | 55.0 | 0 | 472 | 0.004 | ~1/N |

## Interior-answerable coverage (family × N)

*How many confound-clean, interior-only questions exist per length — the power budget for interior geometry-linked behavior.*

| family | N=7 | N=9 | N=12 | N=16 |
|---|--:|--:|--:|--:|
| `betweenness` | 720 | 2160 | 2160 | 2160 |
| `comparative_distance` | 1440 | 2160 | 2160 | 2160 |
| `count_between` | 2160 | 2160 | 2160 | 2160 |
| `pairwise` | 4320 | 7200 | 8640 | 11520 |
| `predecessor` | 1440 | 2160 | 2160 | 2160 |
| `rank` | 1882 | 3218 | 4801 | 5353 |
| `successor` | 1440 | 2160 | 2160 | 2160 |

## Structure × family (question counts)

- **total_order**: `betweenness`=17280, `comparative_distance`=17280, `count_between`=17280, `extremes`=5760, `pairwise`=82080, `predecessor`=14400, `rank`=24480, `reconstruction`=5760, `successor`=14400
- **partial_order**: `order_query`=34560
- **grid2d**: `pairwise`=34560

## rank-distance histograms (integration reach)

*Distance-stratified categories: d=1 is directly stated by a card; d≥2 requires transitive integration (the real difficulty axis).*

- `count_between`: {1: 4424, 2: 3273, 3: 2368, 4: 1853, 5: 1414, 6: 1080, 7: 798, 8: 612, 9: 449, 10: 369, 11: 252, 12: 141, 13: 125, 14: 81, 15: 41}
- `pairwise`: {1: 23040, 2: 12744, 3: 10296, 4: 9154, 5: 6322, 6: 4664, 7: 2900, 8: 5302, 9: 3174, 10: 2094, 11: 1286, 12: 430, 13: 356, 14: 212, 15: 106}
- `predecessor`: {1: 14400}
- `successor`: {1: 14400}
