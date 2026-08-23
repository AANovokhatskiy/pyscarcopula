# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `f0c49d9e9113389b019cac00bb8b2fac1853f166`
- Compute source digest: `c70549c0d543e1a566974e98e78463b76b5b3f98b5c9ab0f3e3dd014d6564a90`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.005117555 | 2.452% | 36.527% |
| `pair.clayton.r0.grid` | 0.033059375 | 2.154% | 5.282% |
| `pair.clayton.r90.grid` | 0.0327449 | 0.404% | 4.231% |
| `pair.clayton.r180.grid` | 0.034007125 | 1.724% | 4.912% |
| `pair.clayton.r270.grid` | 0.033734375 | 2.392% | 3.531% |
| `pair.gumbel.r0.grid` | 0.041538825 | 3.923% | 5.839% |
| `pair.gumbel.r90.grid` | 0.03993705 | 1.249% | 5.174% |
| `pair.gumbel.r180.grid` | 0.040971325 | 4.043% | 5.287% |
| `pair.gumbel.r270.grid` | 0.03977995 | 0.663% | 1.345% |
| `pair.joe.r0.grid` | 0.04461055 | 3.322% | 6.794% |
| `pair.joe.r90.grid` | 0.04303625 | 0.489% | 1.359% |
| `pair.joe.r180.grid` | 0.043287525 | 1.622% | 5.708% |
| `pair.joe.r270.grid` | 0.043046675 | 0.309% | 1.324% |
| `pair.frank.r0.grid` | 0.0315841 | 3.013% | 6.072% |
| `pair.gaussian.r0.grid` | 0.0143215625 | 3.175% | 14.451% |
| `latency.pair.clayton.r0` | 3.71883864e-05 | 0.457% | 4.482% |
| `transform.clayton.softplus` | 0.001944875 | 2.421% | 8.535% |
| `transform.clayton.xtanh` | 0.00134658333 | 2.915% | 14.331% |
| `transform.clayton.exp` | 0.00123312542 | 1.244% | 4.384% |
| `transform.clayton.logistic` | 0.00176728049 | 1.969% | 9.394% |
| `transform.frank.softplus` | 0.00193106029 | 1.259% | 4.341% |
| `transform.frank.xtanh` | 0.00131243047 | 1.469% | 9.751% |
| `transform.frank.exp` | 0.0012312049 | 1.665% | 5.068% |
| `transform.frank.logistic` | 0.00175129054 | 2.048% | 5.511% |
| `transform.gumbel.softplus` | 0.00190921607 | 2.921% | 11.726% |
| `transform.gumbel.xtanh` | 0.00136037059 | 3.162% | 5.042% |
| `transform.gumbel.exp` | 0.00119659265 | 1.338% | 7.036% |
| `transform.gumbel.logistic` | 0.00171249167 | 1.859% | 9.212% |
| `transform.joe.softplus` | 0.00194389737 | 4.085% | 11.936% |
| `transform.joe.xtanh` | 0.00135100082 | 1.394% | 6.328% |
| `transform.joe.exp` | 0.00122119718 | 3.147% | 5.422% |
| `transform.joe.logistic` | 0.00175664394 | 2.391% | 6.015% |
| `static.gaussian.dense.t1` | 0.000465886555 | 0.628% | 7.538% |
| `static.gaussian.dense.tphysical` | 7.93752113e-05 | 2.243% | 4.520% |
| `static.gaussian.factor.t1` | 0.0012731686 | 0.602% | 8.659% |
| `static.gaussian.factor.tphysical` | 0.000159973171 | 1.277% | 8.300% |
| `grid.equicorr.t1` | 0.0168768 | 1.505% | 9.131% |
| `grid.equicorr.t2` | 0.0101586583 | 4.684% | 33.362% |
| `grid.equicorr.t4` | 0.007339625 | 5.978% | 20.434% |
| `grid.equicorr.tphysical` | 0.00441493889 | 3.243% | 21.194% |
| `grid.student.dense.t1` | 0.00987148 | 1.665% | 5.103% |
| `grid.student.dense.t2` | 0.005261305 | 1.644% | 5.859% |
| `grid.student.dense.t4` | 0.003171775 | 3.720% | 12.757% |
| `grid.student.dense.tphysical` | 0.00106709687 | 0.520% | 8.251% |
| `grid.student.factor.t1` | 0.65884405 | 0.372% | 2.143% |
| `grid.student.factor.t2` | 0.3299202 | 0.343% | 5.596% |
| `grid.student.factor.t4` | 0.18921535 | 0.267% | 1.452% |
| `grid.student.factor.tphysical` | 0.04967015 | 3.017% | 12.311% |
| `gas.pair.gumbel` | 0.00066264 | 2.201% | 20.477% |
| `gas.student.dense` | 0.000264474571 | 0.417% | 4.230% |
| `scar_ou.matrix.cold` | 0.000749829661 | 0.479% | 7.093% |
| `scar_ou.matrix.prepared` | 0.000718188182 | 0.583% | 11.261% |
| `scar_ou.local.cold` | 0.000906209677 | 3.547% | 10.013% |
| `scar_ou.local.prepared` | 0.000846473881 | 0.369% | 0.667% |
| `scar_ou.spectral.cold` | 0.00153864444 | 0.302% | 8.011% |
| `scar_ou.spectral.prepared` | 0.00140708205 | 0.307% | 1.562% |
| `vine.density.t1` | 0.00142069394 | 1.307% | 7.322% |
| `vine.density.tphysical` | 0.00124490641 | 1.090% | 2.640% |
| `vine.rosenblatt.t1` | 0.000976488596 | 0.645% | 2.564% |
| `vine.rosenblatt.tphysical` | 0.000868225439 | 1.651% | 8.421% |
| `vine.sampling.t1` | 0.00208100217 | 2.072% | 3.929% |
| `vine.sampling.tphysical` | 0.00184065192 | 2.353% | 8.226% |
| `vine.mcmc.t1` | 0.00299625 | 0.674% | 2.103% |
| `vine.mcmc.tphysical` | 0.00264068158 | 2.110% | 3.212% |
| `conditional.gaussian.t1` | 0.00117918023 | 0.894% | 2.815% |
| `conditional.gaussian.tphysical` | 0.000250166667 | 1.702% | 5.765% |
| `conditional.student.t1` | 0.000508478713 | 0.498% | 1.797% |
| `conditional.student.tphysical` | 0.000129492576 | 2.048% | 8.870% |
