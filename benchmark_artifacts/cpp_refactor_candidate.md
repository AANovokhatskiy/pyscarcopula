# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `384a905c987b54b775516190345f2d07aa2e606a`
- Compute source digest: `037a1d762fae997e274b6afe1c9d2ac8373dcd4a4c68fab0c50cb8ff3aea4360`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00334643438 | 1.092% | 13.309% |
| `pair.clayton.r0.grid` | 0.0140956375 | 1.171% | 14.186% |
| `pair.clayton.r90.grid` | 0.01424525 | 1.017% | 7.245% |
| `pair.clayton.r180.grid` | 0.0143310125 | 0.620% | 1.642% |
| `pair.clayton.r270.grid` | 0.0141138125 | 1.025% | 5.196% |
| `pair.gumbel.r0.grid` | 0.0133254125 | 0.792% | 1.850% |
| `pair.gumbel.r90.grid` | 0.013416325 | 1.008% | 2.160% |
| `pair.gumbel.r180.grid` | 0.0134901 | 1.730% | 3.740% |
| `pair.gumbel.r270.grid` | 0.01331235 | 1.111% | 4.024% |
| `pair.joe.r0.grid` | 0.028926325 | 1.107% | 3.333% |
| `pair.joe.r90.grid` | 0.02958585 | 2.066% | 3.359% |
| `pair.joe.r180.grid` | 0.0294767 | 0.980% | 2.351% |
| `pair.joe.r270.grid` | 0.0289669 | 0.678% | 4.805% |
| `pair.frank.r0.grid` | 0.0228727167 | 0.773% | 5.538% |
| `pair.gaussian.r0.grid` | 0.01108895 | 1.087% | 4.164% |
| `latency.pair.clayton.r0` | 3.37168543e-05 | 1.216% | 3.572% |
| `transform.clayton.softplus` | 0.00197642292 | 0.326% | 1.868% |
| `transform.clayton.xtanh` | 0.00136413636 | 2.918% | 15.685% |
| `transform.clayton.exp` | 0.00126922206 | 2.353% | 26.531% |
| `transform.clayton.logistic` | 0.00183092045 | 2.761% | 5.123% |
| `transform.frank.softplus` | 0.00201876364 | 2.583% | 8.148% |
| `transform.frank.xtanh` | 0.00140107833 | 3.216% | 12.620% |
| `transform.frank.exp` | 0.00125330758 | 2.043% | 9.343% |
| `transform.frank.logistic` | 0.00181947885 | 2.451% | 6.930% |
| `transform.gumbel.softplus` | 0.00200692619 | 1.562% | 8.434% |
| `transform.gumbel.xtanh` | 0.00141788571 | 2.107% | 6.333% |
| `transform.gumbel.exp` | 0.00130235147 | 1.321% | 7.988% |
| `transform.gumbel.logistic` | 0.00181205172 | 2.334% | 5.100% |
| `transform.joe.softplus` | 0.00203808333 | 3.073% | 8.761% |
| `transform.joe.xtanh` | 0.00142822656 | 2.058% | 9.265% |
| `transform.joe.exp` | 0.00127195606 | 2.315% | 11.698% |
| `transform.joe.logistic` | 0.00177565714 | 1.953% | 5.943% |
| `static.gaussian.dense.t1` | 0.00045518314 | 0.474% | 2.039% |
| `static.gaussian.dense.tphysical` | 7.38146845e-05 | 3.052% | 7.602% |
| `static.gaussian.factor.t1` | 0.00127813667 | 0.590% | 1.934% |
| `static.gaussian.factor.tphysical` | 0.000152614241 | 3.184% | 5.952% |
| `grid.equicorr.t1` | 0.0172280333 | 0.431% | 1.802% |
| `grid.equicorr.t2` | 0.0099576 | 1.129% | 3.752% |
| `grid.equicorr.t4` | 0.00727425625 | 1.435% | 2.165% |
| `grid.equicorr.tphysical` | 0.00442933 | 2.459% | 9.382% |
| `grid.student.dense.t1` | 0.0096184 | 0.914% | 3.257% |
| `grid.student.dense.t2` | 0.0050185 | 1.441% | 2.521% |
| `grid.student.dense.t4` | 0.00313471765 | 1.786% | 6.653% |
| `grid.student.dense.tphysical` | 0.00096495 | 1.369% | 9.443% |
| `grid.student.factor.t1` | 0.6597615 | 0.415% | 1.623% |
| `grid.student.factor.t2` | 0.3324128 | 0.823% | 4.177% |
| `grid.student.factor.t4` | 0.19028345 | 0.501% | 1.476% |
| `grid.student.factor.tphysical` | 0.047265125 | 2.245% | 5.501% |
| `gas.pair.gumbel` | 0.000653824405 | 0.535% | 1.669% |
| `gas.student.dense` | 0.00026822551 | 0.820% | 3.272% |
| `scar_ou.matrix.cold` | 0.000758567568 | 0.685% | 2.122% |
| `scar_ou.matrix.prepared` | 0.000738386076 | 0.831% | 2.630% |
| `scar_ou.local.cold` | 0.00091482037 | 0.918% | 4.283% |
| `scar_ou.local.prepared` | 0.000884625758 | 1.124% | 2.060% |
| `scar_ou.spectral.cold` | 0.00163268226 | 1.537% | 6.839% |
| `scar_ou.spectral.prepared` | 0.00144241053 | 0.760% | 2.565% |
| `vine.density.t1` | 0.00151831282 | 1.014% | 2.749% |
| `vine.density.tphysical` | 0.00135505135 | 1.053% | 1.931% |
| `vine.rosenblatt.t1` | 0.00106611136 | 1.352% | 8.439% |
| `vine.rosenblatt.tphysical` | 0.000953823529 | 1.502% | 1.946% |
| `vine.sampling.t1` | 0.00215430682 | 1.045% | 3.523% |
| `vine.sampling.tphysical` | 0.00194766154 | 0.830% | 2.442% |
| `vine.mcmc.t1` | 0.00316724118 | 0.478% | 2.289% |
| `vine.mcmc.tphysical` | 0.00284146389 | 1.725% | 5.502% |
| `conditional.gaussian.t1` | 0.00136695 | 2.371% | 8.280% |
| `conditional.gaussian.tphysical` | 0.000260737037 | 8.971% | 38.136% |
| `conditional.student.t1` | 0.00052675297 | 1.060% | 3.574% |
| `conditional.student.tphysical` | 0.00015024252 | 1.644% | 2.674% |
