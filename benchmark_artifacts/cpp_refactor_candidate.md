# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `1a5a3aba81e9536b655a9a93618ced3e835153b7`
- Compute source digest: `a38d38676a479d123880a054b3bd08457b0cd1b3f99aa77d2eb0ebe760c4221a`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00304167059 | 0.430% | 2.936% |
| `pair.clayton.r0.grid` | 0.0135665 | 0.425% | 3.261% |
| `pair.clayton.r90.grid` | 0.0135736 | 0.373% | 1.361% |
| `pair.clayton.r180.grid` | 0.013636025 | 0.609% | 1.579% |
| `pair.clayton.r270.grid` | 0.01359825 | 0.753% | 1.689% |
| `pair.gumbel.r0.grid` | 0.01272345 | 0.582% | 8.419% |
| `pair.gumbel.r90.grid` | 0.0127310125 | 0.471% | 1.406% |
| `pair.gumbel.r180.grid` | 0.0128788375 | 0.699% | 1.905% |
| `pair.gumbel.r270.grid` | 0.0128507375 | 0.406% | 1.600% |
| `pair.joe.r0.grid` | 0.02803665 | 0.310% | 1.184% |
| `pair.joe.r90.grid` | 0.028114275 | 0.311% | 1.411% |
| `pair.joe.r180.grid` | 0.02827155 | 0.782% | 1.204% |
| `pair.joe.r270.grid` | 0.028080025 | 0.467% | 1.005% |
| `pair.frank.r0.grid` | 0.02201085 | 0.408% | 1.214% |
| `pair.gaussian.r0.grid` | 0.01031819 | 0.369% | 2.602% |
| `latency.pair.clayton.r0` | 3.33663409e-05 | 0.641% | 1.578% |
| `transform.clayton.softplus` | 0.00196333333 | 1.253% | 9.886% |
| `transform.clayton.xtanh` | 0.00132458971 | 1.693% | 6.849% |
| `transform.clayton.exp` | 0.00136658056 | 1.315% | 4.854% |
| `transform.clayton.logistic` | 0.00178495 | 1.768% | 7.054% |
| `transform.frank.softplus` | 0.0019220037 | 1.515% | 5.565% |
| `transform.frank.xtanh` | 0.00134199355 | 2.096% | 9.013% |
| `transform.frank.exp` | 0.00117435658 | 2.576% | 6.678% |
| `transform.frank.logistic` | 0.00177648621 | 1.915% | 6.370% |
| `transform.gumbel.softplus` | 0.00194224259 | 6.383% | 12.444% |
| `transform.gumbel.xtanh` | 0.00150668966 | 1.127% | 3.100% |
| `transform.gumbel.exp` | 0.00138697639 | 0.601% | 2.893% |
| `transform.gumbel.logistic` | 0.00168881731 | 1.225% | 3.652% |
| `transform.joe.softplus` | 0.00196677143 | 1.610% | 4.534% |
| `transform.joe.xtanh` | 0.00139659583 | 2.115% | 5.340% |
| `transform.joe.exp` | 0.00118994412 | 3.161% | 6.943% |
| `transform.joe.logistic` | 0.001900218 | 0.663% | 2.977% |
| `static.gaussian.dense.t1` | 0.000444868548 | 0.540% | 4.089% |
| `static.gaussian.dense.tphysical` | 6.99701872e-05 | 1.780% | 9.045% |
| `static.gaussian.factor.t1` | 0.00123443375 | 0.582% | 3.575% |
| `static.gaussian.factor.tphysical` | 0.000169736364 | 1.729% | 5.154% |
| `grid.equicorr.t1` | 0.0167863667 | 2.336% | 11.379% |
| `grid.equicorr.t2` | 0.00948901 | 1.692% | 6.315% |
| `grid.equicorr.t4` | 0.00693648 | 3.010% | 13.136% |
| `grid.equicorr.tphysical` | 0.00416932778 | 2.102% | 19.240% |
| `grid.student.dense.t1` | 0.00943228333 | 0.571% | 1.281% |
| `grid.student.dense.t2` | 0.00489248636 | 1.244% | 2.879% |
| `grid.student.dense.t4` | 0.00299087353 | 1.972% | 6.408% |
| `grid.student.dense.tphysical` | 0.000882680233 | 1.606% | 4.567% |
| `grid.student.factor.t1` | 0.64668275 | 0.133% | 0.432% |
| `grid.student.factor.t2` | 0.32362785 | 0.649% | 1.552% |
| `grid.student.factor.t4` | 0.18819605 | 0.346% | 0.595% |
| `grid.student.factor.tphysical` | 0.046948075 | 1.735% | 6.846% |
| `gas.pair.gumbel` | 0.000628585256 | 0.433% | 1.050% |
| `gas.student.dense` | 0.000253051683 | 0.689% | 0.941% |
| `scar_ou.matrix.cold` | 0.000738205333 | 0.674% | 1.234% |
| `scar_ou.matrix.prepared` | 0.000707342361 | 0.550% | 1.372% |
| `scar_ou.local.cold` | 0.000868085156 | 0.389% | 1.189% |
| `scar_ou.local.prepared` | 0.000831207547 | 0.328% | 1.352% |
| `scar_ou.spectral.cold` | 0.00149799595 | 0.257% | 1.077% |
| `scar_ou.spectral.prepared` | 0.00138249512 | 0.309% | 1.336% |
| `vine.density.t1` | 0.00141810938 | 0.446% | 1.732% |
| `vine.density.tphysical` | 0.00130268676 | 1.296% | 8.649% |
| `vine.rosenblatt.t1` | 0.000961407018 | 0.600% | 3.901% |
| `vine.rosenblatt.tphysical` | 0.000878583673 | 1.225% | 3.909% |
| `vine.sampling.t1` | 0.002007162 | 0.269% | 1.204% |
| `vine.sampling.tphysical` | 0.00187515357 | 1.858% | 7.560% |
| `vine.mcmc.t1` | 0.00301858529 | 0.283% | 1.551% |
| `vine.mcmc.tphysical` | 0.002699875 | 1.440% | 9.249% |
| `conditional.gaussian.t1` | 0.000957177119 | 0.459% | 1.124% |
| `conditional.gaussian.tphysical` | 0.000212640857 | 1.189% | 9.553% |
| `conditional.student.t1` | 0.000505620089 | 0.538% | 2.341% |
| `conditional.student.tphysical` | 0.000145731569 | 1.024% | 1.939% |
