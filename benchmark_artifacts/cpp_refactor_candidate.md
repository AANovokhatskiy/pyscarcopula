# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `7e59ddb97da9032e6bf136b408ab586190b9e29b`
- Compute source digest: `b9acebb9965ec1f204fe7bd9572325b3def2a7d4d1a480d5b643f82d711a66b7`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00482286923 | 4.366% | 30.188% |
| `pair.clayton.r0.grid` | 0.0152859375 | 0.830% | 3.317% |
| `pair.clayton.r90.grid` | 0.0153453833 | 0.968% | 3.148% |
| `pair.clayton.r180.grid` | 0.01618425 | 5.016% | 8.351% |
| `pair.clayton.r270.grid` | 0.015267325 | 1.494% | 3.565% |
| `pair.gumbel.r0.grid` | 0.014378025 | 1.045% | 1.968% |
| `pair.gumbel.r90.grid` | 0.0143211375 | 0.470% | 2.165% |
| `pair.gumbel.r180.grid` | 0.0148011375 | 0.821% | 1.280% |
| `pair.gumbel.r270.grid` | 0.0142905125 | 0.409% | 4.744% |
| `pair.joe.r0.grid` | 0.03148535 | 0.952% | 2.789% |
| `pair.joe.r90.grid` | 0.0319251 | 1.388% | 3.881% |
| `pair.joe.r180.grid` | 0.03145595 | 1.420% | 5.521% |
| `pair.joe.r270.grid` | 0.031378975 | 0.670% | 2.828% |
| `pair.frank.r0.grid` | 0.02469195 | 1.706% | 3.612% |
| `pair.gaussian.r0.grid` | 0.0120071625 | 0.903% | 4.248% |
| `latency.pair.clayton.r0` | 4.12064718e-05 | 0.414% | 2.464% |
| `transform.clayton.softplus` | 0.002056225 | 2.867% | 9.286% |
| `transform.clayton.xtanh` | 0.00137429375 | 7.026% | 24.825% |
| `transform.clayton.exp` | 0.00111918537 | 6.353% | 24.897% |
| `transform.clayton.logistic` | 0.0018279119 | 2.953% | 12.123% |
| `transform.frank.softplus` | 0.00210235577 | 2.030% | 4.096% |
| `transform.frank.xtanh` | 0.00124475526 | 1.146% | 7.408% |
| `transform.frank.exp` | 0.00124244375 | 10.165% | 26.378% |
| `transform.frank.logistic` | 0.00196472963 | 2.374% | 5.072% |
| `transform.gumbel.softplus` | 0.00215302879 | 2.050% | 6.297% |
| `transform.gumbel.xtanh` | 0.00140535431 | 2.385% | 4.170% |
| `transform.gumbel.exp` | 0.00118928382 | 1.253% | 9.174% |
| `transform.gumbel.logistic` | 0.00179972614 | 2.119% | 5.524% |
| `transform.joe.softplus` | 0.00194397895 | 1.192% | 4.585% |
| `transform.joe.xtanh` | 0.00130668676 | 2.825% | 5.847% |
| `transform.joe.exp` | 0.0011606007 | 1.888% | 3.962% |
| `transform.joe.logistic` | 0.00179238387 | 1.402% | 5.045% |
| `static.gaussian.dense.t1` | 0.000496616532 | 0.426% | 5.322% |
| `static.gaussian.dense.tphysical` | 7.42980357e-05 | 2.739% | 12.865% |
| `static.gaussian.factor.t1` | 0.00137986842 | 0.599% | 1.630% |
| `static.gaussian.factor.tphysical` | 0.000140767598 | 3.121% | 9.207% |
| `grid.equicorr.t1` | 0.0187839333 | 0.413% | 2.785% |
| `grid.equicorr.t2` | 0.01012914 | 0.918% | 5.205% |
| `grid.equicorr.t4` | 0.00690540625 | 1.201% | 4.209% |
| `grid.equicorr.tphysical` | 0.00417668333 | 1.530% | 2.264% |
| `grid.student.dense.t1` | 0.01045092 | 1.532% | 1.483% |
| `grid.student.dense.t2` | 0.005191495 | 0.871% | 3.754% |
| `grid.student.dense.t4` | 0.00310920312 | 1.957% | 8.841% |
| `grid.student.dense.tphysical` | 0.000880810417 | 1.048% | 7.372% |
| `grid.student.factor.t1` | 0.71277085 | 0.505% | 2.219% |
| `grid.student.factor.t2` | 0.34776205 | 0.667% | 1.962% |
| `grid.student.factor.t4` | 0.1896198 | 0.476% | 1.663% |
| `grid.student.factor.tphysical` | 0.045204675 | 1.317% | 5.446% |
| `gas.pair.gumbel` | 0.000733033929 | 1.710% | 2.239% |
| `gas.student.dense` | 0.000292646875 | 1.400% | 4.237% |
| `scar_ou.matrix.cold` | 0.00082962 | 0.341% | 1.308% |
| `scar_ou.matrix.prepared` | 0.000782231707 | 0.774% | 2.679% |
| `scar_ou.local.cold` | 0.000975670192 | 0.412% | 0.936% |
| `scar_ou.local.prepared` | 0.000954325373 | 0.543% | 2.367% |
| `scar_ou.spectral.cold` | 0.00169006346 | 0.537% | 1.455% |
| `scar_ou.spectral.prepared` | 0.0015332425 | 0.464% | 1.801% |
| `vine.density.t1` | 0.00162281852 | 0.256% | 0.389% |
| `vine.density.tphysical` | 0.00129621667 | 0.189% | 0.950% |
| `vine.rosenblatt.t1` | 0.00117751087 | 0.469% | 1.862% |
| `vine.rosenblatt.tphysical` | 0.000909745455 | 0.274% | 1.237% |
| `vine.sampling.t1` | 0.00233515238 | 0.962% | 2.533% |
| `vine.sampling.tphysical` | 0.00186937407 | 0.620% | 1.941% |
| `vine.mcmc.t1` | 0.00338157692 | 0.658% | 2.188% |
| `vine.mcmc.tphysical` | 0.00267237368 | 0.611% | 1.412% |
| `conditional.gaussian.t1` | 0.00107064583 | 0.614% | 2.106% |
| `conditional.gaussian.tphysical` | 0.000200710853 | 1.245% | 9.494% |
| `conditional.student.t1` | 0.000559241593 | 0.448% | 2.114% |
| `conditional.student.tphysical` | 0.000122683012 | 0.733% | 7.592% |
