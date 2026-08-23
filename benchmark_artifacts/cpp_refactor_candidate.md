# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `0c30123d8790e3a9f526e72406618b207361e1e4`
- Compute source digest: `c492bb11b5f231e91b52c860fd54e934288a9bd46a2fc7a4d2c30722a235d9ea`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00503581818 | 1.077% | 2.263% |
| `pair.clayton.r0.grid` | 0.032618025 | 0.719% | 1.270% |
| `pair.clayton.r90.grid` | 0.032994775 | 0.958% | 8.960% |
| `pair.clayton.r180.grid` | 0.032807875 | 0.694% | 1.242% |
| `pair.clayton.r270.grid` | 0.032676575 | 0.392% | 5.105% |
| `pair.gumbel.r0.grid` | 0.039674275 | 0.433% | 1.240% |
| `pair.gumbel.r90.grid` | 0.0400035 | 1.534% | 4.128% |
| `pair.gumbel.r180.grid` | 0.0395013 | 0.310% | 1.642% |
| `pair.gumbel.r270.grid` | 0.04015905 | 1.752% | 5.162% |
| `pair.joe.r0.grid` | 0.042823775 | 0.232% | 0.527% |
| `pair.joe.r90.grid` | 0.043191075 | 2.008% | 5.509% |
| `pair.joe.r180.grid` | 0.043324075 | 0.719% | 1.018% |
| `pair.joe.r270.grid` | 0.043213475 | 1.173% | 6.114% |
| `pair.frank.r0.grid` | 0.031121825 | 0.588% | 6.553% |
| `pair.gaussian.r0.grid` | 0.014266075 | 0.608% | 1.537% |
| `latency.pair.clayton.r0` | 3.76632281e-05 | 2.014% | 3.735% |
| `transform.clayton.softplus` | 0.00185835208 | 1.829% | 9.257% |
| `transform.clayton.xtanh` | 0.0012679875 | 2.356% | 6.036% |
| `transform.clayton.exp` | 0.0011521413 | 1.361% | 5.777% |
| `transform.clayton.logistic` | 0.00169766562 | 2.636% | 8.134% |
| `transform.frank.softplus` | 0.001871735 | 0.964% | 2.848% |
| `transform.frank.xtanh` | 0.00124409143 | 1.204% | 4.630% |
| `transform.frank.exp` | 0.00114586 | 1.725% | 7.965% |
| `transform.frank.logistic` | 0.00168773676 | 2.182% | 7.611% |
| `transform.gumbel.softplus` | 0.00187247679 | 1.961% | 3.134% |
| `transform.gumbel.xtanh` | 0.00130311 | 1.522% | 12.266% |
| `transform.gumbel.exp` | 0.00115651667 | 2.082% | 8.986% |
| `transform.gumbel.logistic` | 0.00172846034 | 2.358% | 8.273% |
| `transform.joe.softplus` | 0.00187368333 | 0.996% | 3.646% |
| `transform.joe.xtanh` | 0.00126774605 | 2.874% | 13.961% |
| `transform.joe.exp` | 0.00124764623 | 2.794% | 9.773% |
| `transform.joe.logistic` | 0.00166894074 | 2.323% | 5.602% |
| `static.gaussian.dense.t1` | 0.00046425625 | 0.854% | 12.055% |
| `static.gaussian.dense.tphysical` | 7.69255814e-05 | 1.170% | 13.762% |
| `static.gaussian.factor.t1` | 0.00127594857 | 0.767% | 2.608% |
| `static.gaussian.factor.tphysical` | 0.000158659276 | 2.169% | 4.934% |
| `grid.equicorr.t1` | 0.0168608833 | 0.634% | 1.211% |
| `grid.equicorr.t2` | 0.01000275 | 1.824% | 8.774% |
| `grid.equicorr.t4` | 0.0073205875 | 2.944% | 5.026% |
| `grid.equicorr.tphysical` | 0.00460025 | 1.768% | 5.758% |
| `grid.student.dense.t1` | 0.00995685 | 0.481% | 1.573% |
| `grid.student.dense.t2` | 0.005350445 | 2.203% | 5.556% |
| `grid.student.dense.t4` | 0.00313117333 | 1.634% | 5.448% |
| `grid.student.dense.tphysical` | 0.00112372889 | 1.632% | 7.118% |
| `grid.student.factor.t1` | 0.6582732 | 0.725% | 16.741% |
| `grid.student.factor.t2` | 0.32893765 | 0.580% | 2.019% |
| `grid.student.factor.t4` | 0.1902414 | 0.616% | 1.268% |
| `grid.student.factor.tphysical` | 0.0518781 | 4.037% | 11.455% |
| `gas.pair.gumbel` | 0.000631253012 | 0.944% | 7.796% |
| `gas.student.dense` | 0.000264634821 | 0.584% | 1.634% |
| `scar_ou.matrix.cold` | 0.000744560135 | 0.912% | 394.480% |
| `scar_ou.matrix.prepared` | 0.000710942949 | 0.443% | 1.741% |
| `scar_ou.local.cold` | 0.000877664167 | 0.402% | 2.599% |
| `scar_ou.local.prepared` | 0.000853408209 | 1.842% | 4.031% |
| `scar_ou.spectral.cold` | 0.00152486833 | 0.384% | 1.240% |
| `scar_ou.spectral.prepared` | 0.00138095143 | 0.435% | 1.277% |
| `vine.density.t1` | 0.00136941857 | 0.563% | 13.219% |
| `vine.density.tphysical` | 0.00123590854 | 0.672% | 3.858% |
| `vine.rosenblatt.t1` | 0.0009622125 | 0.272% | 1.516% |
| `vine.rosenblatt.tphysical` | 0.000874880208 | 3.077% | 5.206% |
| `vine.sampling.t1` | 0.002037022 | 1.363% | 5.747% |
| `vine.sampling.tphysical` | 0.001848125 | 1.807% | 5.705% |
| `vine.mcmc.t1` | 0.00303795313 | 0.508% | 2.399% |
| `vine.mcmc.tphysical` | 0.002581725 | 0.627% | 9.375% |
| `conditional.gaussian.t1` | 0.000989672917 | 0.783% | 1.946% |
| `conditional.gaussian.tphysical` | 0.00022179717 | 5.753% | 13.522% |
| `conditional.student.t1` | 0.000512564602 | 1.520% | 2.762% |
| `conditional.student.tphysical` | 0.000147929596 | 2.848% | 7.652% |
