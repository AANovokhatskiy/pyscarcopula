# C++ refactor benchmark capture

- Manifest: `cpp-architecture-refactor-v1`
- Commit: `0c30123d8790e3a9f526e72406618b207361e1e4`
- Compute source digest: `c492bb11b5f231e91b52c860fd54e934288a9bd46a2fc7a4d2c30722a235d9ea`
- Cases: 68
- Valid for regression check: True
- Validity: eligible capture under the declared regression protocol

Percentage noise metrics below are diagnostic and never block a change.

| Case | Median, s | relMAD | Pair noise |
|---|---:|---:|---:|
| `pair.independent.r0.grid` | 0.00516110455 | 2.027% | 15.661% |
| `pair.clayton.r0.grid` | 0.0326034 | 0.408% | 6.006% |
| `pair.clayton.r90.grid` | 0.03271765 | 0.797% | 8.637% |
| `pair.clayton.r180.grid` | 0.0327407 | 0.990% | 5.824% |
| `pair.clayton.r270.grid` | 0.0326094 | 0.444% | 1.196% |
| `pair.gumbel.r0.grid` | 0.0398252 | 0.978% | 5.157% |
| `pair.gumbel.r90.grid` | 0.0397416 | 0.260% | 1.163% |
| `pair.gumbel.r180.grid` | 0.039619675 | 0.413% | 4.350% |
| `pair.gumbel.r270.grid` | 0.03987145 | 0.505% | 0.715% |
| `pair.joe.r0.grid` | 0.042828675 | 0.402% | 1.219% |
| `pair.joe.r90.grid` | 0.04296385 | 0.751% | 3.804% |
| `pair.joe.r180.grid` | 0.04303535 | 0.803% | 2.509% |
| `pair.joe.r270.grid` | 0.0432598 | 1.761% | 6.235% |
| `pair.frank.r0.grid` | 0.031005275 | 0.733% | 2.280% |
| `pair.gaussian.r0.grid` | 0.0142104833 | 0.573% | 5.341% |
| `latency.pair.clayton.r0` | 3.78062807e-05 | 1.594% | 6.436% |
| `transform.clayton.softplus` | 0.00203605227 | 2.405% | 8.954% |
| `transform.clayton.xtanh` | 0.00121676905 | 1.263% | 5.655% |
| `transform.clayton.exp` | 0.00109378308 | 2.148% | 7.278% |
| `transform.clayton.logistic` | 0.00163831667 | 2.369% | 3.801% |
| `transform.frank.softplus` | 0.00182238906 | 1.173% | 4.588% |
| `transform.frank.xtanh` | 0.00122103026 | 0.911% | 8.888% |
| `transform.frank.exp` | 0.00113015221 | 3.696% | 13.927% |
| `transform.frank.logistic` | 0.0016309803 | 1.752% | 5.059% |
| `transform.gumbel.softplus` | 0.00182626184 | 2.023% | 4.907% |
| `transform.gumbel.xtanh` | 0.00122005294 | 2.397% | 8.554% |
| `transform.gumbel.exp` | 0.00109082273 | 1.689% | 5.759% |
| `transform.gumbel.logistic` | 0.00166946094 | 2.187% | 6.959% |
| `transform.joe.softplus` | 0.00187919423 | 2.229% | 4.585% |
| `transform.joe.xtanh` | 0.001227572 | 2.896% | 6.346% |
| `transform.joe.exp` | 0.00114748977 | 2.437% | 7.846% |
| `transform.joe.logistic` | 0.00164822353 | 1.996% | 5.038% |
| `static.gaussian.dense.t1` | 0.000464465882 | 0.439% | 2.727% |
| `static.gaussian.dense.tphysical` | 7.45329327e-05 | 2.696% | 9.262% |
| `static.gaussian.factor.t1` | 0.00127425909 | 0.240% | 1.106% |
| `static.gaussian.factor.tphysical` | 0.000160308659 | 3.805% | 6.492% |
| `grid.equicorr.t1` | 0.0171484833 | 3.405% | 8.920% |
| `grid.equicorr.t2` | 0.01000798 | 4.066% | 12.118% |
| `grid.equicorr.t4` | 0.00727980833 | 2.462% | 17.896% |
| `grid.equicorr.tphysical` | 0.00451245625 | 1.865% | 17.626% |
| `grid.student.dense.t1` | 0.00977845833 | 0.576% | 2.127% |
| `grid.student.dense.t2` | 0.005154425 | 1.538% | 9.280% |
| `grid.student.dense.t4` | 0.00318460937 | 2.075% | 6.369% |
| `grid.student.dense.tphysical` | 0.00104333191 | 0.832% | 4.333% |
| `grid.student.factor.t1` | 0.65965125 | 0.282% | 1.926% |
| `grid.student.factor.t2` | 0.3298813 | 0.405% | 2.313% |
| `grid.student.factor.t4` | 0.18905695 | 0.351% | 1.017% |
| `grid.student.factor.tphysical` | 0.049475675 | 1.427% | 5.048% |
| `gas.pair.gumbel` | 0.000634411798 | 0.568% | 3.102% |
| `gas.student.dense` | 0.000267050246 | 0.587% | 1.503% |
| `scar_ou.matrix.cold` | 0.000751865714 | 0.333% | 1.531% |
| `scar_ou.matrix.prepared` | 0.000713042157 | 0.577% | 1.696% |
| `scar_ou.local.cold` | 0.000882329688 | 0.212% | 1.324% |
| `scar_ou.local.prepared` | 0.000843501064 | 0.339% | 1.363% |
| `scar_ou.spectral.cold` | 0.0015441431 | 0.626% | 1.465% |
| `scar_ou.spectral.prepared` | 0.00140921897 | 0.387% | 1.420% |
| `vine.density.t1` | 0.00141052353 | 2.282% | 10.445% |
| `vine.density.tphysical` | 0.00124272429 | 0.840% | 4.396% |
| `vine.rosenblatt.t1` | 0.000974607895 | 0.293% | 6.119% |
| `vine.rosenblatt.tphysical` | 0.000880975893 | 1.777% | 10.468% |
| `vine.sampling.t1` | 0.00205136522 | 0.626% | 9.113% |
| `vine.sampling.tphysical` | 0.00183843929 | 1.650% | 6.202% |
| `vine.mcmc.t1` | 0.00300686176 | 0.282% | 6.833% |
| `vine.mcmc.tphysical` | 0.00263134333 | 0.561% | 2.846% |
| `conditional.gaussian.t1` | 0.00143962973 | 3.417% | 10.179% |
| `conditional.gaussian.tphysical` | 0.00023824 | 11.217% | 43.528% |
| `conditional.student.t1` | 0.000509177027 | 1.162% | 1.233% |
| `conditional.student.tphysical` | 0.000130272995 | 1.923% | 9.231% |
