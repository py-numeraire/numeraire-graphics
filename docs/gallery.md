# Gallery

Every figure, grouped by the input family it consumes. Each is a pure function returning a `ggplot`;
compose it with {func}`~numeraire_graphics.theme_numeraire` and a colour scale, then render or save.

## Family A — result-schema plotters

These read the tidy result table every evaluator emits — one row per `method × date × metric`.
**Per-date** rows (e.g. `StrategyReturnEvaluator`'s `strategy_return`) feed the time-series plots;
**summary** rows (e.g. `sharpe`) feed the bar and curve plots.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} plot_cumulative
Geometric wealth curve with a drawdown facet, an optional benchmark line and recession shading.
```python
plot_cumulative(results, benchmark=None, recessions=None)
```
:::

:::{grid-item-card} plot_rolling
Trailing-window rolling Sharpe, mean, or volatility — one line per method.
```python
plot_rolling(results, window=36, metric="sharpe")
```
:::

:::{grid-item-card} plot_metric_by
A summary metric as bars across a grouping column, with confidence-interval whiskers when derivable.
```python
plot_metric_by(results, metric="sharpe", x="method")
```
:::

:::{grid-item-card} plot_complexity_curve
A metric plotted against a caller-supplied complexity axis (shrinkage intensity, parameter count).
```python
plot_complexity_curve(results, x="n_params", metric="oos_r2", ribbon=None)
```
:::

:::{grid-item-card} plot_ic_decay
Information-coefficient decay by horizon, from `ICEvaluator` rows joined to a numeric horizon.
```python
plot_ic_decay(results, horizon="horizon", metric="ic", smooth=False)
```
:::

::::

## Family B — Output / frame plotters

These need inputs the tidy schema does not carry, so they take a numeraire Output object or a
caller-supplied frame directly.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} plot_weights_heatmap
A date × asset weight matrix as `geom_tile`, signed long/short, from a
{class}`~numeraire.WeightsOutput` object.
```python
plot_weights_heatmap(weights_output, top=None, order="mean")
```
:::

:::{grid-item-card} plot_factor_loadings
Loading paths over an axis, or a loadings heatmap, from a caller-supplied tidy loadings frame.
```python
plot_factor_loadings(loadings, x="date")
```
:::

:::{grid-item-card} plot_frontier
A risk–return efficient frontier, optionally overlaying named portfolios, from a `risk`/`return`
frame.
```python
plot_frontier(frontier, points=None)
```
:::

::::

## Helpers

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} theme_numeraire
The publication house theme (serif, flat strips, y-only grid) — compose it onto any figure.
```python
fig + theme_numeraire(base_size=8)
```
:::

:::{grid-item-card} scale_color_numeraire / scale_fill_numeraire
Colourblind-safe Okabe–Ito discrete scales; the fill scale also offers a zero-centred diverging mode
for the signed weights heatmap.
```python
fig + scale_color_numeraire()
fig + scale_fill_numeraire(diverging=True)
```
:::

:::{grid-item-card} save_paper
The sole save surface — sizes the figure exactly in centimetres under a print font profile.
```python
save_paper(fig, "figure.pdf", width_cm=8, height_cm=6)
```
:::

:::{grid-item-card} mean_variance_frontier
A small numpy-only convenience that traces a `risk`/`return` frame from a mean vector and covariance
for `plot_frontier`.
```python
frontier = mean_variance_frontier(mean, cov, n=50)
```
:::

::::
