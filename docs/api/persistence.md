# Persistence API

Fitted models can be saved to JSON and restored without depending on Python
object pickling. Set `include_data=True` when stateful prediction should use
the training history stored by the fitted model.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pyscarcopula import GumbelCopula, load_model, save_model

rng = np.random.default_rng(2026)
source = GumbelCopula(rotate=180)
u = source.sample_at_parameter(200, np.full(200, 1.7), rng=rng)

model = GumbelCopula(rotate=180)
model.fit(u, method="mle")

with TemporaryDirectory() as directory:
    path = Path(directory) / "gumbel.json"
    save_model(model, path, include_data=True)
    restored = load_model(path)
    samples = restored.predict(20, rng=np.random.default_rng(7))
```

Model instances also expose `model.save(...)`, and model classes provide a
matching `load(...)` convenience method.

For Equicorr models fitted to `EquicorrPreparedData`, `include_data=True`
retains the compact sufficient statistics, without reconstructing the original
observations. Loading restores their validation and read-only arrays.
Dataclass records encoded with the `object` tag are rejected; they must use
the `dataclass` tag so that loading invokes their validating constructor.
`include_data=False` omits both dense training observations and prepared
statistics; fitted parameters and diagnostics are retained. Saving does not
change the source model's retained history.

::: pyscarcopula.io.save_model

::: pyscarcopula.io.load_model
