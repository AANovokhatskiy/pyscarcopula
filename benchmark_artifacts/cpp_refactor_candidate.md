# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `eca69b55d4a6333c867bbc2ec8fd8a4b83f1cb4e`
- Compute source digest: `565c03c460b1dd505bbe6de31b7bfc62c14bf2b2117b301598e46dec3e932c9b`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.0050545 | 19.334% | 53.888% |
| `pair.clayton.r0.grid` | 0.0290552 | 2.650% | 14.593% |
| `pair.clayton.r90.grid` | 0.02759705 | 3.258% | 16.859% |
| `pair.clayton.r180.grid` | 0.02851715 | 2.723% | 56.854% |
| `pair.clayton.r270.grid` | 0.0307275 | 6.486% | 18.855% |
| `pair.gumbel.r0.grid` | 0.035448525 | 3.795% | 38.469% |
| `pair.gumbel.r90.grid` | 0.0378904 | 6.891% | 32.280% |
| `pair.gumbel.r180.grid` | 0.03582135 | 4.648% | 5.137% |
| `pair.gumbel.r270.grid` | 0.034921225 | 4.041% | 8.655% |
| `pair.joe.r0.grid` | 0.041239125 | 9.039% | 10.782% |
| `pair.joe.r90.grid` | 0.039733725 | 3.290% | 52.790% |
| `pair.joe.r180.grid` | 0.04111525 | 5.967% | 20.038% |
| `pair.joe.r270.grid` | 0.03944115 | 3.718% | 37.683% |
| `pair.frank.r0.grid` | 0.0257527667 | 7.627% | 50.809% |
| `pair.gaussian.r0.grid` | 0.0116738875 | 5.887% | 55.259% |
| `latency.pair.clayton.r0` | 3.59438148e-05 | 5.934% | 74.928% |
| `transform.clayton.softplus` | 0.00207487143 | 2.742% | 10.378% |
| `transform.clayton.xtanh` | 0.0015741 | 8.829% | 46.586% |
| `transform.clayton.exp` | 0.00129582 | 3.419% | 15.748% |
| `transform.clayton.logistic` | 0.00202789655 | 6.783% | 76.445% |
| `transform.frank.softplus` | 0.00223551 | 11.177% | 26.833% |
| `transform.frank.xtanh` | 0.00157987564 | 4.016% | 25.882% |
| `transform.frank.exp` | 0.00138247703 | 3.780% | 11.799% |
| `transform.frank.logistic` | 0.001909292 | 7.248% | 15.795% |
| `transform.gumbel.softplus` | 0.002092954 | 1.156% | 3.771% |
| `transform.gumbel.xtanh` | 0.00152763594 | 3.121% | 5.466% |
| `transform.gumbel.exp` | 0.00130542375 | 2.273% | 5.753% |
| `transform.gumbel.logistic` | 0.00188300833 | 20.379% | 34.183% |
| `transform.joe.softplus` | 0.00190313621 | 5.823% | 21.772% |
| `transform.joe.xtanh` | 0.00134400571 | 4.066% | 11.972% |
| `transform.joe.exp` | 0.00127789362 | 7.447% | 23.993% |
| `transform.joe.logistic` | 0.00174528846 | 9.717% | 16.528% |
| `static.gaussian.dense.t1` | 0.000454368443 | 2.016% | 91.456% |
| `static.gaussian.dense.tphysical` | 8.2028479e-05 | 7.917% | 13.447% |
| `static.gaussian.factor.t1` | 0.00138810256 | 14.806% | 23.635% |
| `static.gaussian.factor.tphysical` | 0.000162085021 | 2.302% | 6.711% |
| `grid.equicorr.t1` | 0.0185225125 | 9.351% | 57.923% |
| `grid.equicorr.t2` | 0.01049717 | 3.359% | 13.766% |
| `grid.equicorr.t4` | 0.00704710833 | 5.843% | 12.044% |
| `grid.equicorr.tphysical` | 0.00399967857 | 0.853% | 2.108% |
| `grid.student.dense.t1` | 0.01104913 | 9.692% | 208.769% |
| `grid.student.dense.t2` | 0.00571851 | 1.111% | 17.559% |
| `grid.student.dense.t4` | 0.00333021 | 3.395% | 8.525% |
| `grid.student.dense.tphysical` | 0.00110615244 | 1.008% | 3.868% |
| `grid.student.factor.t1` | 0.64911895 | 3.182% | 6.807% |
| `grid.student.factor.t2` | 0.33104725 | 1.914% | 17.969% |
| `grid.student.factor.t4` | 0.1924816 | 1.895% | 24.275% |
| `grid.student.factor.tphysical` | 0.04827645 | 4.577% | 22.154% |
| `gas.pair.gumbel` | 0.000629252809 | 5.914% | 18.627% |
| `gas.student.dense` | 0.000241587562 | 1.080% | 25.510% |
| `scar_ou.matrix.cold` | 0.000717947101 | 6.084% | 18.153% |
| `scar_ou.matrix.prepared` | 0.000704303797 | 3.028% | 21.884% |
| `scar_ou.local.cold` | 0.000814514844 | 4.145% | 5.909% |
| `scar_ou.local.prepared` | 0.000780511538 | 1.097% | 4.888% |
| `scar_ou.spectral.cold` | 0.00149648387 | 0.580% | 3.384% |
| `scar_ou.spectral.prepared` | 0.00137587436 | 3.682% | 23.874% |
| `vine.density.t1` | 0.00162287969 | 6.822% | 31.346% |
| `vine.density.tphysical` | 0.00162270517 | 1.486% | 10.576% |
| `vine.rosenblatt.t1` | 0.000941237719 | 2.107% | 5.203% |
| `vine.rosenblatt.tphysical` | 0.00116428953 | 0.650% | 3.282% |
| `vine.sampling.t1` | 0.00195574464 | 1.979% | 7.421% |
| `vine.sampling.tphysical` | 0.002156416 | 1.573% | 3.627% |
| `vine.mcmc.t1` | 0.00298125294 | 1.176% | 7.151% |
| `vine.mcmc.tphysical` | 0.0034177875 | 1.738% | 3.813% |
| `conditional.gaussian.t1` | 0.00095026413 | 0.797% | 6.880% |
| `conditional.gaussian.tphysical` | 0.000242683577 | 2.180% | 11.103% |
| `conditional.student.t1` | 0.000490998571 | 1.051% | 3.627% |
| `conditional.student.tphysical` | 0.00015004375 | 1.561% | 8.894% |
