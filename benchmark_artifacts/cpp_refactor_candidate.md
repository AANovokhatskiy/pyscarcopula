# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `None`
- Compute source digest: `e0b60ce0ec9f79a72a735fd67d7cfa66c69401763e39658308886118ea59ca3b`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00318613437 | 2.042% | 6.261% |
| `pair.clayton.r0.grid` | 0.013851225 | 0.585% | 2.124% |
| `pair.clayton.r90.grid` | 0.013870975 | 0.541% | 1.436% |
| `pair.clayton.r180.grid` | 0.0139714375 | 1.365% | 5.214% |
| `pair.clayton.r270.grid` | 0.013877525 | 1.110% | 5.182% |
| `pair.gumbel.r0.grid` | 0.0130145125 | 0.506% | 2.544% |
| `pair.gumbel.r90.grid` | 0.013046125 | 0.566% | 2.596% |
| `pair.gumbel.r180.grid` | 0.0131580375 | 1.082% | 2.583% |
| `pair.gumbel.r270.grid` | 0.0131488625 | 2.378% | 3.818% |
| `pair.joe.r0.grid` | 0.028482025 | 0.915% | 7.123% |
| `pair.joe.r90.grid` | 0.02864805 | 0.879% | 3.693% |
| `pair.joe.r180.grid` | 0.029166375 | 2.231% | 2.916% |
| `pair.joe.r270.grid` | 0.028589025 | 1.199% | 3.979% |
| `pair.frank.r0.grid` | 0.02234385 | 0.366% | 3.885% |
| `pair.gaussian.r0.grid` | 0.01056305 | 0.405% | 2.744% |
| `latency.pair.clayton.r0` | 3.46244785e-05 | 0.723% | 4.280% |
| `transform.clayton.softplus` | 0.00197047619 | 1.395% | 11.801% |
| `transform.clayton.xtanh` | 0.0013877375 | 0.726% | 2.904% |
| `transform.clayton.exp` | 0.00125069595 | 2.938% | 9.360% |
| `transform.clayton.logistic` | 0.00164579 | 0.585% | 1.515% |
| `transform.frank.softplus` | 0.00185230312 | 1.233% | 5.777% |
| `transform.frank.xtanh` | 0.00120470106 | 1.350% | 5.949% |
| `transform.frank.exp` | 0.00107199118 | 2.083% | 3.981% |
| `transform.frank.logistic` | 0.00158582703 | 0.682% | 4.699% |
| `transform.gumbel.softplus` | 0.00180413485 | 1.121% | 2.306% |
| `transform.gumbel.xtanh` | 0.00120502813 | 1.368% | 4.224% |
| `transform.gumbel.exp` | 0.00109927411 | 1.251% | 5.976% |
| `transform.gumbel.logistic` | 0.00158825 | 1.841% | 4.566% |
| `transform.joe.softplus` | 0.00183162794 | 1.776% | 3.680% |
| `transform.joe.xtanh` | 0.00121421765 | 1.141% | 3.508% |
| `transform.joe.exp` | 0.00106635821 | 0.653% | 8.097% |
| `transform.joe.logistic` | 0.00158955909 | 1.950% | 3.864% |
| `static.gaussian.dense.t1` | 0.000455636458 | 0.906% | 3.084% |
| `static.gaussian.dense.tphysical` | 7.55200837e-05 | 4.176% | 10.099% |
| `static.gaussian.factor.t1` | 0.00127721951 | 0.821% | 3.036% |
| `static.gaussian.factor.tphysical` | 0.000149170433 | 2.755% | 11.832% |
| `grid.equicorr.t1` | 0.0166376833 | 0.511% | 1.773% |
| `grid.equicorr.t2` | 0.009637275 | 1.155% | 4.526% |
| `grid.equicorr.t4` | 0.0070695125 | 2.207% | 4.963% |
| `grid.equicorr.tphysical` | 0.00420515909 | 1.898% | 4.929% |
| `grid.student.dense.t1` | 0.00941745833 | 0.440% | 1.374% |
| `grid.student.dense.t2` | 0.00498933 | 0.694% | 2.216% |
| `grid.student.dense.t4` | 0.00306977059 | 2.154% | 6.420% |
| `grid.student.dense.tphysical` | 0.000952661702 | 5.475% | 11.526% |
| `grid.student.factor.t1` | 0.65688995 | 0.093% | 0.924% |
| `grid.student.factor.t2` | 0.3278776 | 0.629% | 1.452% |
| `grid.student.factor.t4` | 0.1891682 | 0.552% | 1.945% |
| `grid.student.factor.tphysical` | 0.046101575 | 2.425% | 7.998% |
| `gas.pair.gumbel` | 0.000654721154 | 0.518% | 3.009% |
| `gas.student.dense` | 0.000259709746 | 0.866% | 1.916% |
| `scar_ou.matrix.cold` | 0.000747376 | 0.344% | 2.539% |
| `scar_ou.matrix.prepared` | 0.000716305405 | 0.167% | 0.988% |
| `scar_ou.local.cold` | 0.000884930769 | 0.759% | 1.612% |
| `scar_ou.local.prepared` | 0.000851585821 | 0.683% | 2.002% |
| `scar_ou.spectral.cold` | 0.00151767273 | 0.406% | 1.128% |
| `scar_ou.spectral.prepared` | 0.00139866429 | 0.263% | 1.126% |
| `vine.density.t1` | 0.00146040735 | 0.443% | 2.856% |
| `vine.density.tphysical` | 0.00132279189 | 1.455% | 10.809% |
| `vine.rosenblatt.t1` | 0.00104114778 | 0.969% | 3.027% |
| `vine.rosenblatt.tphysical` | 0.000932346154 | 1.892% | 7.334% |
| `vine.sampling.t1` | 0.00211504783 | 0.597% | 2.269% |
| `vine.sampling.tphysical` | 0.00190251296 | 1.430% | 11.298% |
| `vine.mcmc.t1` | 0.00312342647 | 0.572% | 1.120% |
| `vine.mcmc.tphysical` | 0.00269403158 | 0.688% | 2.953% |
| `conditional.gaussian.t1` | 0.000973928571 | 0.316% | 1.983% |
| `conditional.gaussian.tphysical` | 0.000211598266 | 0.840% | 7.269% |
| `conditional.student.t1` | 0.000507985185 | 0.409% | 1.368% |
| `conditional.student.tphysical` | 0.000133515203 | 5.290% | 5.957% |
