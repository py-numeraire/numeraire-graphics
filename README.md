# numeraire-viz

Grammar-of-graphics figures over the [**numeraire**](https://github.com/py-numeraire/numeraire)
tidy result schema.

**plotnine is primary.** Every plot function *returns* a `ggplot` grammar object — it never draws,
shows, or saves. You compose the returned object freely (`+ theme_numeraire()` and friends) and,
when you are ready, hand it to the one explicit save surface, `save_paper`, for centimetre-exact
figures. Core `numeraire` stays visualization-free; this is a separate, optional package.

## Install

```bash
pip install numeraire-viz          # pulls in numeraire, plotnine, matplotlib, mizani
```

## The result schema

Everything here consumes numeraire's tidy result table (the columns every evaluator emits):

```
run_id, method, date, metric, value, universe, capability, protocol, config_hash, data_vintage
```

Two idioms drive the figures: **per-date** rows (one row per date per method — e.g.
`StrategyReturnEvaluator` emits `metric="strategy_return"`) feed the time-series plots, and
**summary** rows (one scalar row per method — e.g. `metric="sharpe"`) feed the bar/curve plots.

## Plots (first slice)

| function | reads | draws |
|----------|-------|-------|
| `plot_cumulative(results, *, benchmark=None, recessions=None)` | `metric == "strategy_return"` per-date rows | geometric wealth curve + drawdown facets, optional dashed benchmark line and `geom_rect` recession shading (dates supplied by the caller) |
| `plot_rolling(results, *, window, metric="sharpe")` | the same per-date returns | trailing-`window` rolling Sharpe / mean / vol, one line per method |
| `plot_metric_by(results, *, metric, x="method")` | a scalar summary `metric` | bars across a grouping column, with CI whiskers when a confidence interval is derivable |
| `plot_complexity_curve(results, *, x, metric, ribbon=None)` | a scalar `metric` + a caller-joined numeric `x` | metric-vs-complexity curve with an optional ribbon band |

`x` in `plot_complexity_curve` (a shrinkage intensity, parameter count, ...) is **not** part of the
result schema — you join it onto the frame yourself and name the column; the function will not
invent it.

## Helpers

- `theme_numeraire(base_size=8, base_family="serif")` — the house theme (publication-oriented).
- `scale_color_numeraire(palette="okabe_ito", greyscale=False)` — a colourblind-safe discrete
  colour scale (Okabe-Ito by default; `greyscale=True` for monochrome print, paired with linetypes).
- `save_paper(plot, path, *, width_cm, height_cm, font_profile="latex")` — the sole save surface;
  sizes the figure exactly in centimetres under a print font profile.

```python
from numeraire_viz import plot_cumulative, theme_numeraire, scale_color_numeraire, save_paper

fig = plot_cumulative(results) + theme_numeraire() + scale_color_numeraire()
save_paper(fig, "cumulative.pdf", width_cm=8, height_cm=6)
```

## Conventions

Okabe-Ito palette by default, greyscale-safe linetypes, zero-reference lines, percent axis labels
where the quantity is a return. Figure captions belong in the LaTeX document, not baked into the
figure.

## Roadmap (not in this slice)

- **More plots**: weights heatmap, factor-loading paths, efficient frontier, IC-decay fan — each
  needs inputs beyond the result schema (a weight stream, a loadings panel, a frontier trace, an
  IC-by-horizon table), so they wait on the corresponding numeraire surfaces.
- **`[altair]` extra**: a narwhals-native exploration surface (tooltips, selection, HTML sharing).
- **`[tables]` extra**: publication tables (great_tables) companion to the figures.

License: BSD-3-Clause.
