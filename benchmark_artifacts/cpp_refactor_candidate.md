# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `b082428db43c5aa1565b72a6e17316a130130222`
- Compute source digest: `ea413546c6496b622624f97f0b95821b29f5b183d627073ba94faaa98ac76b1f`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol
- Comparison passed: True
- Comparison failures: 0

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00420597083 | 0.420% | 2.428% |
| `pair.clayton.r0.grid` | 0.0297022 | 0.296% | 0.917% |
| `pair.clayton.r90.grid` | 0.030521475 | 2.350% | 19.516% |
| `pair.clayton.r180.grid` | 0.0302543 | 0.320% | 1.902% |
| `pair.clayton.r270.grid` | 0.030124275 | 0.405% | 1.328% |
| `pair.gumbel.r0.grid` | 0.037024875 | 1.419% | 2.023% |
| `pair.gumbel.r90.grid` | 0.036291475 | 0.372% | 0.871% |
| `pair.gumbel.r180.grid` | 0.038753725 | 3.019% | 6.327% |
| `pair.gumbel.r270.grid` | 0.038491425 | 0.697% | 15.288% |
| `pair.joe.r0.grid` | 0.03959595 | 0.970% | 1.223% |
| `pair.joe.r90.grid` | 0.03987 | 0.539% | 1.201% |
| `pair.joe.r180.grid` | 0.039756875 | 0.409% | 4.050% |
| `pair.joe.r270.grid` | 0.0400548 | 0.513% | 1.917% |
| `pair.frank.r0.grid` | 0.028145925 | 0.593% | 22.855% |
| `pair.gaussian.r0.grid` | 0.0127915 | 1.018% | 19.926% |
| `latency.pair.clayton.r0` | 3.39342105e-05 | 0.389% | 1.142% |
| `transform.clayton.softplus` | 0.00172885556 | 0.909% | 3.584% |
| `transform.clayton.xtanh` | 0.00121659886 | 0.440% | 2.495% |
| `transform.clayton.exp` | 0.00110425 | 1.273% | 1.893% |
| `transform.clayton.logistic` | 0.00157424032 | 0.723% | 5.326% |
| `transform.frank.softplus` | 0.0017562569 | 0.140% | 0.769% |
| `transform.frank.xtanh` | 0.00131155795 | 6.637% | 15.833% |
| `transform.frank.exp` | 0.00134139024 | 9.124% | 33.622% |
| `transform.frank.logistic` | 0.00160245606 | 1.566% | 6.855% |
| `transform.gumbel.softplus` | 0.00189547931 | 6.897% | 17.223% |
| `transform.gumbel.xtanh` | 0.001314475 | 1.743% | 18.137% |
| `transform.gumbel.exp` | 0.00121320875 | 0.794% | 2.859% |
| `transform.gumbel.logistic` | 0.0017670431 | 5.000% | 27.285% |
| `transform.joe.softplus` | 0.00179246296 | 3.651% | 8.510% |
| `transform.joe.xtanh` | 0.0012146122 | 0.944% | 14.007% |
| `transform.joe.exp` | 0.0010985 | 0.635% | 3.401% |
| `transform.joe.logistic` | 0.00157988077 | 2.447% | 8.310% |
| `static.gaussian.dense.t1` | 0.00043715 | 1.365% | 11.521% |
| `static.gaussian.dense.tphysical` | 8.27011811e-05 | 2.237% | 7.613% |
| `static.gaussian.factor.t1` | 0.001174375 | 0.880% | 29.567% |
| `static.gaussian.factor.tphysical` | 0.000151129767 | 4.117% | 11.666% |
| `grid.equicorr.t1` | 0.015634125 | 1.740% | 2.150% |
| `grid.equicorr.t2` | 0.00927691667 | 1.508% | 5.729% |
| `grid.equicorr.t4` | 0.0076531375 | 1.686% | 4.788% |
| `grid.equicorr.tphysical` | 0.00381224615 | 1.616% | 5.729% |
| `grid.student.dense.t1` | 0.00986305833 | 9.836% | 16.844% |
| `grid.student.dense.t2` | 0.00441882143 | 1.048% | 38.448% |
| `grid.student.dense.t4` | 0.00359762667 | 3.755% | 10.792% |
| `grid.student.dense.tphysical` | 0.00118634865 | 8.124% | 20.776% |
| `grid.student.factor.t1` | 0.61422205 | 2.010% | 11.017% |
| `grid.student.factor.t2` | 0.3068064 | 0.879% | 3.966% |
| `grid.student.factor.t4` | 0.1934602 | 0.260% | 0.921% |
| `grid.student.factor.tphysical` | 0.04380695 | 0.331% | 2.057% |
| `gas.pair.gumbel` | 0.000581420253 | 1.391% | 4.197% |
| `gas.student.dense` | 0.000239373881 | 0.254% | 1.131% |
| `scar_ou.matrix.cold` | 0.000685403676 | 0.250% | 1.129% |
| `scar_ou.matrix.prepared` | 0.000666538608 | 0.422% | 2.091% |
| `scar_ou.local.cold` | 0.000827570968 | 1.982% | 1.836% |
| `scar_ou.local.prepared` | 0.000791184848 | 0.441% | 1.311% |
| `scar_ou.spectral.cold` | 0.00141966892 | 0.606% | 1.690% |
| `scar_ou.spectral.prepared` | 0.00131665625 | 0.634% | 2.392% |
| `vine.density.t1` | 0.00134778611 | 2.282% | 31.375% |
| `vine.density.tphysical` | 0.00138058906 | 7.831% | 9.615% |
| `vine.rosenblatt.t1` | 0.000935903509 | 0.362% | 0.941% |
| `vine.rosenblatt.tphysical` | 0.00114802727 | 0.411% | 1.310% |
| `vine.sampling.t1` | 0.00184733846 | 0.360% | 1.051% |
| `vine.sampling.tphysical` | 0.00204954167 | 3.332% | 5.756% |
| `vine.mcmc.t1` | 0.00272135263 | 0.488% | 1.004% |
| `vine.mcmc.tphysical` | 0.00334030937 | 0.593% | 9.591% |
| `conditional.gaussian.t1` | 0.000895008475 | 0.515% | 1.619% |
| `conditional.gaussian.tphysical` | 0.000240571622 | 9.504% | 28.119% |
| `conditional.student.t1` | 0.000462581818 | 0.897% | 4.345% |
| `conditional.student.tphysical` | 0.000138158803 | 0.867% | 6.842% |
