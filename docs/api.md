# API reference

Every public name in `numeraire_graphics`. Plot functions return a `ggplot`; helpers compose onto a
figure or save it. See the {doc}`gallery` for what each figure looks like.

```{eval-rst}
.. currentmodule:: numeraire_graphics

.. rubric:: Plots

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_cumulative
   plot_rolling
   plot_metric_by
   plot_complexity_curve
   plot_ic_decay
   plot_weights_heatmap
   plot_factor_loadings
   plot_frontier

.. rubric:: Themes, scales, and saving

.. autosummary::
   :toctree: generated
   :nosignatures:

   theme_numeraire
   scale_color_numeraire
   scale_fill_numeraire
   save_paper
   mean_variance_frontier
```
