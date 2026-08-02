# BCS benchmark strata

*Generated from `data/bcs` — analysis-ready breakdown of every query category. Interior-answerable = all involved entities in ranks 3..N−2 (the confound-clean subset used for geometry-linked behavior). Endpoint-diagnostic = leaks role/frequency, reported as a control only.*

- families: `s0_zib, s1_size, s1_loud, s1_heat`  ·  N-grid: `[7, 9, 12, 16, 24]`  ·  per-cell: 100  ·  difficulty: both  ·  degree: 4
- gate_failures at generation: **0**

- **stimuli:** 14400  {'total_order': 8000, 'partial_order': 1600, 'grid2d': 1600, 'cyclic': 3200}
- **coherence-null twins:** 4000
- **questions:** 408000  (orphans: 0)
- **degeneracy flags:** none

## Per-category strata

| family | n | interior | interior % | endpoint | #keys | top-key share | chance |
|---|--:|--:|--:|--:|--:|--:|---|
| `betweenness` | 24000 | 10400 | 43.3 | 0 | 472 | 0.003 | 0.33 (3-choice) |
| `comparative_distance` | 24000 | 11200 | 46.7 | 0 | 472 | 0.004 | 0.50 (2-choice) |
| `count_between` | 24000 | 12000 | 50.0 | 0 | 23 | 0.22 | const-0 ~26%; report by rank_distance / MAE, not 1/(N-1) |
| `cyclic_distance` | 12800 | 0 | 0.0 | 0 | 23 | 0.088 | ~1/N (integer 0..N-1) |
| `cyclic_order` | 12800 | 0 | 0.0 | 0 | 472 | 0.004 | 0.50 (2-choice) |
| `cyclic_predecessor` | 12800 | 0 | 0.0 | 0 | 472 | 0.004 | ~1/N |
| `cyclic_successor` | 12800 | 0 | 0.0 | 0 | 472 | 0.003 | ~1/N |
| `extremes` | 8000 | 0 | 0.0 | 8000 | 472 | 0.004 | ~1/N (endpoint-diagnostic) |
| `order_query` | 38400 | 0 | 0.0 | 0 | 473 | 0.5 | 0.50 (2-choice + 'undetermined') |
| `pairwise` | 155200 | 48000 | 30.9 | 0 | 472 | 0.003 | 0.50 (2-choice) |
| `predecessor` | 20000 | 11200 | 56.0 | 0 | 472 | 0.004 | ~1/N |
| `rank` | 35200 | 23591 | 67.0 | 5829 | 496 | 0.043 | ~1/N |
| `reconstruction` | 8000 | 0 | 0.0 | 0 | 2001 | 0.5 | Kendall tau = 0 |
| `successor` | 20000 | 11200 | 56.0 | 0 | 472 | 0.003 | ~1/N |

## Interior-answerable coverage (family × N)

*How many confound-clean, interior-only questions exist per length — the power budget for interior geometry-linked behavior.*

| family | N=7 | N=9 | N=12 | N=16 | N=24 |
|---|--:|--:|--:|--:|--:|
| `betweenness` | 800 | 2400 | 2400 | 2400 | 2400 |
| `comparative_distance` | 1600 | 2400 | 2400 | 2400 | 2400 |
| `count_between` | 2400 | 2400 | 2400 | 2400 | 2400 |
| `pairwise` | 4800 | 8000 | 9600 | 12800 | 12800 |
| `predecessor` | 1600 | 2400 | 2400 | 2400 | 2400 |
| `rank` | 2079 | 3576 | 5322 | 5946 | 6668 |
| `successor` | 1600 | 2400 | 2400 | 2400 | 2400 |

## Structure × family (question counts)

- **total_order**: `betweenness`=24000, `comparative_distance`=24000, `count_between`=24000, `extremes`=8000, `pairwise`=116800, `predecessor`=20000, `rank`=35200, `reconstruction`=8000, `successor`=20000
- **partial_order**: `order_query`=38400
- **grid2d**: `pairwise`=38400
- **cyclic**: `cyclic_distance`=12800, `cyclic_order`=12800, `cyclic_predecessor`=12800, `cyclic_successor`=12800

## rank-distance histograms (integration reach)

*Distance-stratified categories: d=1 is directly stated by a card; d≥2 requires transitive integration (the real difficulty axis).*

- `count_between`: {1: 5283, 2: 3997, 3: 2908, 4: 2405, 5: 1855, 6: 1460, 7: 1177, 8: 958, 9: 729, 10: 642, 11: 512, 12: 383, 13: 314, 14: 278, 15: 244, 16: 150, 17: 148, 18: 136, 19: 130, 20: 127, 21: 82, 22: 55, 23: 27}
- `pairwise`: {1: 32000, 2: 17188, 3: 14812, 4: 11808, 5: 8656, 6: 6800, 7: 4736, 8: 6610, 9: 4130, 10: 3042, 11: 1952, 12: 1056, 13: 922, 14: 664, 15: 550, 16: 384, 17: 306, 18: 284, 19: 310, 20: 248, 21: 166, 22: 114, 23: 62}
- `predecessor`: {1: 20000}
- `successor`: {1: 20000}
