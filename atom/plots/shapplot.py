# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modeling (ATOM)
Author: Mavs
Description: Module containing the ShapPlot class.

"""

from __future__ import annotations

from abc import ABC
from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
import shap
from beartype import beartype
from beartype.typing import Any, Hashable

from atom.plots.baseplot import BasePlot
from atom.utils.types import (
    Bool, Int, IntLargerZero, Legend, ModelSelector, RowSelector,
    TargetsSelector,
)
from atom.utils.utils import check_canvas, crash


@beartype
class ShapPlot(BasePlot, ABC):
    """Shap plots.

    ATOM wrapper for plots made by the shap package, using Shapley
    values for model interpretation. These plots are accessible from
    the runners or from the models. Only one model can be plotted at
    the same time since the plots are not made by ATOM.

    """

    @crash
    def plot_shap_bar(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's bar plot.

        Create a bar plot of a set of SHAP values. If a single sample
        is passed, then the SHAP values are plotted. If many samples
        are passed, then the mean absolute value for each feature
        column is plotted. Read more about SHAP plots in the
        [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_bar()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_bar(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_bar")

        shap.plots.bar(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_bar",
            filename=filename,
            display=display,
        )

    @crash
    def plot_shap_beeswarm(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's beeswarm plot.

        The plot is colored by feature values. Read more about SHAP
        plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_beeswarm()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot. The
            plot_shap_beeswarm method does not support plotting a single
            sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:PredictionPlot.plot_parshap
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_beeswarm(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_beeswarm")

        shap.plots.beeswarm(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            filename=filename,
            display=display,
        )

    @crash
    def plot_shap_decision(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's decision plot.

        Visualize model decisions using cumulative SHAP values. Each
        plotted line explains a single model prediction. If a single
        prediction is plotted, feature values are printed in the
        plot (if supplied). If multiple predictions are plotted
        together, feature values will not be printed. Plotting too
        many predictions together will make the plot unintelligible.
        Read more about SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_decision()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_decision(show=10)
        atom.plot_shap_decision(index=-1, show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_decision")

        shap.decision_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=X.columns,
            feature_display_range=slice(-1, -show_c - 1, -1),
            auto_size_plot=False,
            show=False,
        )

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_decision",
            filename=filename,
            display=display,
        )

    @crash
    def plot_shap_force(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 300),
        filename: str | Path | None = None,
        display: Bool | None = True,
        **kwargs,
    ) -> plt.Figure | None:
        """Plot SHAP's force plot.

        Visualize the given SHAP values with an additive force layout.
        Note that by default this plot will render using javascript.
        For a regular figure use `matplotlib=True` (this option is
        only available when only a single sample is plotted). Read more
        about SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_force()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=(900, 300)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure
            (only if `matplotlib=True` in `kwargs`).

        **kwargs
            Additional keyword arguments for [shap.plots.force][force].

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_scatter
        atom.plots:ShapPlot.plot_shap_decision

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_force(index=-2, matplotlib=True, figsize=(1800, 300))
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(create_figure=False, backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_force")

        plot = shap.force_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=X.columns,
            show=False,
            **kwargs,
        )

        if kwargs.get("matplotlib"):
            BasePlot._fig.used_models.append(models_c)
            return self._plot(
                fig=plt.gcf(),
                ax=plt.gca(),
                title=title,
                legend=legend,
                figsize=figsize,
                plotname="plot_shap_force",
                filename=filename,
                display=display,
            )
        else:
            if filename:  # Save to an html file
                if (path := Path(filename)).suffix != ".html":
                    path = path.with_suffix(".html")
                shap.save_html(str(path), plot)
            if display and find_spec("IPython"):
                from IPython.display import display as ipydisplay

                shap.initjs()
                ipydisplay(plot)

            return None

    @crash
    def plot_shap_heatmap(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's heatmap plot.

        This plot is designed to show the population substructure of a
        dataset using supervised clustering and a heatmap. Supervised
        clustering involves clustering data points not by their original
        feature values but by their explanations. Read more about SHAP
        plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_heatmap()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot. The
            plot_shap_heatmap method does not support plotting a single
            sample.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_decision
        atom.plots:ShapPlot.plot_shap_force
        atom.plots:ShapPlot.plot_shap_waterfall

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_heatmap(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_heatmap")

        shap.plots.heatmap(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_heatmap",
            filename=filename,
            display=display,
        )

    @crash
    def plot_shap_scatter(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        columns: Int | str = 0,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's scatter plot.

        Plots the value of the feature on the x-axis and the SHAP value
        of the same feature on the y-axis. This shows how the model
        depends on the given feature, and is like a richer extension of
        the classical partial dependence plots. Vertical dispersion of
        the data points represents interaction effects. Read more about
        SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_scatter()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot. The
            plot_shap_scatter method does not support plotting a single
            sample.

        columns: int or str, default=0
            Column to plot.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_decision
        atom.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_scatter(columns="symmetry error")
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        columns_c = models_c.branch._get_columns(columns, include_target=False)[0]
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        # Get explanation for a specific column
        explanation = explanation[:, models_c.branch.columns.get_loc(columns_c)]

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_scatter")

        shap.plots.scatter(explanation, color=explanation, ax=plt.gca(), show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            ylabel=plt.gca().get_ylabel(),
            title=title,
            legend=legend,
            plotname="plot_shap_scatter",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @crash
    def plot_shap_waterfall(
        self,
        models: ModelSelector | None = None,
        rows: Hashable = 0,
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plot SHAP's waterfall plot.

        The SHAP value of a feature represents the impact of the
        evidence provided by that feature on the model’s output. The
        waterfall plot is designed to visually display how the SHAP
        values (evidence) of each feature move the model output from
        our prior expectation under the background data distribution,
        to the final model prediction given the evidence of all the
        features. Features are sorted by the magnitude of their SHAP
        values with the smallest magnitude features grouped together
        at the bottom of the plot when the number of features in the
        models exceeds the `show` parameter. Read more about SHAP plots
        in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Model to plot. If None, all models are selected. Note that
            leaving the default option could raise an exception if there
            are multiple models. To avoid this, call the plot directly
            from a model, e.g., `atom.lr.plot_shap_waterfall()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Selection of rows][row-and-column-selection] to plot. The
            plot_shap_waterfall method does not support plotting
            multiple samples.

        show: int or None, default=None
            Number of features (ordered by importance) to show. If
            None, it shows all features.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of features shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as png. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [plt.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:ShapPlot.plot_shap_bar
        atom.plots:ShapPlot.plot_shap_beeswarm
        atom.plots:ShapPlot.plot_shap_heatmap

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.run("LR")
        atom.plot_shap_waterfall(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        if len(row := models_c.branch._get_rows(rows)) > 1:
            raise ValueError(
                f"Invalid value for the rows parameter, got {rows}. "
                "The plot_shap_waterfall method does not support "
                f"plotting multiple samples, got {len(row)}."
            )

        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(row, target_c)

        # Waterfall accepts only one row
        explanation.values = explanation.values[0]
        explanation.data = explanation.data[0]

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_waterfall")

        shap.plots.waterfall(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_waterfall",
            filename=filename,
            display=display,
        )
