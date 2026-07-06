# numeraire-graphics

[![PyPI](https://img.shields.io/pypi/v/numeraire-graphics.svg)](https://pypi.org/project/numeraire-graphics/)
[![Python versions](https://img.shields.io/pypi/pyversions/numeraire-graphics.svg)](https://pypi.org/project/numeraire-graphics/)
[![CI](https://github.com/py-numeraire/numeraire-graphics/actions/workflows/ci.yml/badge.svg)](https://github.com/py-numeraire/numeraire-graphics/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Grammar-of-graphics figures over [**numeraire**](https://github.com/py-numeraire/numeraire)'s
results and Output objects. Part of the
[numeraire ecosystem](https://py-numeraire.org/ecosystem.html).

**plotnine is primary.** Every plot function *returns* a `ggplot` grammar object — it never draws,
shows, or saves. You compose the returned object freely (`+ theme_numeraire()` and friends) and,
when you are ready, hand it to the one explicit save surface, `save_paper`, for centimetre-exact
figures. Core `numeraire` stays visualization-free; this is a separate, optional package.

## Two input families

The plots divide by **what they consume**:

- **Family A — result-schema plotters** (`numeraire_graphics.plots`) read the tidy result table every
  evaluator emits (the columns below). They are the default surface: comparison figures assembled
  straight from a run's results.
- **Family B — Output/frame plotters** (`numeraire_graphics.outputs`) need richer inputs the tidy schema
  deliberately does not carry — a per-date × asset weight stream, a factor-loadings panel, a
  risk-return frontier trace. They take a numeraire **Output object** (e.g. a `WeightsOutput`) or a
  caller-supplied **frame** directly. The contract is otherwise identical (return a `ggplot`, never
  draw or save); only the input differs.

The split is principled, not ad hoc: the tidy result schema has one row per `(method, date, metric)`
with a single scalar `value` and no asset/factor/frontier axis, so a figure that needs that axis
cannot be schema-fed and is a family-B plotter by construction.

## Install

```bash
pip install numeraire-graphics          # pulls in numeraire, plotnine, matplotlib, mizani
```

## The result schema

Everything here consumes numeraire's tidy result table (the columns every evaluator emits):

```
run_id, method, date, metric, value, universe, capability, protocol, config_hash, data_vintage
```

Two idioms drive the figures: **per-date** rows (one row per date per method — e.g.
`StrategyReturnEvaluator` emits `metric="strategy_return"`) feed the time-series plots, and
**summary** rows (one scalar row per method — e.g. `metric="sharpe"`) feed the bar/curve plots.

## Family A — result-schema plotters (`numeraire_graphics.plots`)

| function | reads | draws |
|----------|-------|-------|
| `plot_cumulative(results, *, benchmark=None, recessions=None)` | `metric == "strategy_return"` per-date rows | geometric wealth curve + drawdown facets, optional dashed benchmark line and `geom_rect` recession shading (dates supplied by the caller) |
| `plot_rolling(results, *, window, metric="sharpe")` | the same per-date returns | trailing-`window` rolling Sharpe / mean / vol, one line per method |
| `plot_metric_by(results, *, metric, x="method")` | a scalar summary `metric` | bars across a grouping column, with CI whiskers when a confidence interval is derivable |
| `plot_complexity_curve(results, *, x, metric, ribbon=None)` | a scalar `metric` + a caller-joined numeric `x` | metric-vs-complexity curve with an optional ribbon band |
| `plot_ic_decay(results, *, horizon="horizon", metric="ic", smooth=False)` | the `ic` rows (`ICEvaluator`) + a caller-joined numeric `horizon` | information-coefficient decay curve by method, over a zero line, optional linear smooth |

`x` in `plot_complexity_curve` (a shrinkage intensity, parameter count, ...) and `horizon` in
`plot_ic_decay` are **not** part of the result schema — you join the column onto the frame yourself
and name it; the function will not invent it. For the IC decay curve you assemble the axis by running
forecasts at several horizons, running `ICEvaluator` on each `ForecastOutput`, and tagging every
resulting `ic` row with its numeric horizon before stacking the frames.

## Family B — Output/frame plotters (`numeraire_graphics.outputs`)

| function | consumes | draws |
|----------|----------|-------|
| `plot_weights_heatmap(weights_output, *, top=None, order="mean")` | a `WeightsOutput` / `PanelWeightsOutput` **object** | a date × asset weight matrix as `geom_tile`, signed long/short (compose with `scale_fill_numeraire(diverging=True)`); `top` keeps the N largest-average-\|weight\| names, `order` sorts the asset axis |
| `plot_factor_loadings(loadings, *, x=None)` | a caller-supplied tidy loadings frame (`factor`, `loading`, an axis like `date`/`entity`) | loading paths over `x` faceted/coloured by factor, or a loadings heatmap when `x` is absent |
| `plot_frontier(frontier, *, points=None)` | a caller-supplied `risk`/`return` frame | the efficient-frontier curve, optionally overlaying named portfolios (`points` with `risk`/`return`/`label`) as labelled markers |

There is **no** standard core surface for a loadings panel (it is method-local — an IPCA Γ, a
rolling-beta panel) or a frontier trace, so those two take frames directly. The weights heatmap does
consume a first-class core object (`WeightsOutput` / `PanelWeightsOutput` from
`numeraire.core.engine`). `mean_variance_frontier(mean, cov, *, n=50)` is a small numpy-only
convenience (not core, not an optimizer) that traces a `risk`/`return` frame from a mean vector and
covariance for callers who lack a frontier of their own.

## Helpers

- `theme_numeraire(base_size=8, base_family="serif")` — the house theme (publication-oriented).
- `scale_color_numeraire(palette="okabe_ito", greyscale=False)` — a colourblind-safe discrete
  colour scale (Okabe-Ito by default; `greyscale=True` for monochrome print, paired with linetypes).
- `scale_fill_numeraire(palette="okabe_ito", greyscale=False, diverging=False)` — the `fill`
  counterpart: a discrete Okabe-Ito fill for grouped bars (`plot_metric_by`), or, with
  `diverging=True`, a continuous zero-centred fill for the signed weights heatmap (blue = short,
  vermillion = long).
- `save_paper(plot, path, *, width_cm, height_cm, font_profile="latex")` — the sole save surface;
  sizes the figure exactly in centimetres under a print font profile.

```python
from numeraire_graphics import plot_cumulative, theme_numeraire, scale_color_numeraire, save_paper

fig = plot_cumulative(results) + theme_numeraire() + scale_color_numeraire()
save_paper(fig, "cumulative.pdf", width_cm=8, height_cm=6)
```

## Conventions

Okabe-Ito palette by default, greyscale-safe linetypes, zero-reference lines, percent axis labels
where the quantity is a return. Figure captions belong in the LaTeX document, not baked into the
figure.

## Roadmap

- **`[altair]` extra**: a narwhals-native exploration surface (tooltips, selection, HTML sharing).
- **`[tables]` extra**: publication tables (great_tables) companion to the figures.
- **First-class loadings**: `plot_factor_loadings` and `plot_frontier` are frame-fed because
  numeraire has no standard loadings/frontier surface today; if a core loadings accessor lands
  (e.g. a Γ panel accessor), the loadings plot can consume the object directly like the weights
  heatmap does.

## Citation

`numeraire-graphics` is a companion to `numeraire`. If you use the ecosystem in your research, please
cite `numeraire` — see [How to cite](https://github.com/py-numeraire/numeraire#how-to-cite).

License: BSD-3-Clause.
