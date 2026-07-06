---
title: numeraire-graphics
---

# numeraire-graphics

Publication-ready, grammar-of-graphics figures for the [numeraire](https://numeraire.py-numeraire.org/)
research framework. Every plot is a pure function `(results, *, ...) -> ggplot` built with
[plotnine](https://plotnine.org/): it consumes the tidy result schema every evaluator emits, or an
engine Output object such as a {class}`~numeraire.WeightsOutput`, and returns a grammar object. It
**never draws or saves for you** — you compose the result freely and render it yourself, so the
figure stays a value you can theme, facet, or hand to the one explicit save surface.

```python
from numeraire_graphics import plot_cumulative, theme_numeraire, save_paper

fig = plot_cumulative(results) + theme_numeraire()
save_paper(fig, "wealth.pdf", width_cm=8, height_cm=6)   # centimetre-exact, print-ready
```

## Two input families

The plots divide by **what they consume**. **Result-schema plotters** read the tidy result table
(one row per `method × date × metric`) — the default surface for comparison figures assembled
straight from a run. **Output/frame plotters** need richer inputs the tidy schema does not carry — a
per-date × asset weight stream, a loadings panel, a frontier trace — so they take a numeraire Output
object (a {class}`~numeraire.WeightsOutput`) or a caller-supplied frame directly. The contract is
identical either way: return a `ggplot`, never draw or save. See the {doc}`gallery` for every figure.

## Where to go next

- {doc}`installation` — install and the optional extras.
- {doc}`gallery` — every figure, grouped by input family, with the call that produces it.
- {doc}`api` — the full API reference.

```{toctree}
:hidden:

installation
gallery
api
changelog
```
