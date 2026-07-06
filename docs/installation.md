# Installation

```bash
pip install numeraire-graphics
```

This pulls in [`numeraire`](https://numeraire.py-numeraire.org/) (the framework whose results it plots) and
[plotnine](https://plotnine.org/) with its rendering stack. Python 3.11+ is required.

`numeraire-graphics` is a companion package; the framework installs it for you through the
convenience extra:

```bash
pip install "numeraire[graphics]"   # numeraire + numeraire-graphics
pip install "numeraire[all]"        # numeraire + graphics + dataset
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add numeraire-graphics
```

## Verify

```python
from importlib.metadata import version

import numeraire_graphics  # noqa: F401

print(version("numeraire-graphics"))
```
