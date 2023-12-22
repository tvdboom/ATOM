"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the DataPlot class.

"""

from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from beartype import beartype
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from scipy import stats
from sklearn.base import is_classifier

from atom.plots.baseplot import BasePlot
from atom.utils.constants import PALETTE
from atom.utils.types import (
    Bool, ColumnSelector, DataFrame, Int, IntLargerZero, Legend, RowSelector,
    Segment, Sequence, Series,
)
from atom.utils.utils import (
    check_dependency, crash, divide, get_corpus, lst, replace_missing, rnd,
)


@beartype
class DataPlot(BasePlot, metaclass=ABCMeta):
    """Data plots.

    Plots used for understanding and interpretation of the dataset.
    They are only accessible from atom since the other runners should
    be used for model training only, not for data manipulation.

    """

    @crash
    def plot_components(
        self,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the explained variance ratio per component.

        Kept components are colored and discarded components are
        transparent. This plot is available only when feature selection
        was applied with strategy="pca".

        Parameters
        ----------
        show: int or None, default=None
            Number of components to show. None to show all.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of components shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_pca
        atom.plots:DataPlot.plot_rfecv

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("pca", n_features=5)
        atom.plot_components(show=10)
        ```

        """
        if not hasattr(self, "pca_"):
            raise PermissionError(
                "The plot_pca method is only available for instances "
                "that ran feature selection using the 'pca' strategy, "
                "e.g., atom.feature_selection(strategy='pca')."
            )

        # Get the variance ratio per component
        variance = np.array(self.pca_.explained_variance_ratio_)

        show_c = self._get_show(show, len(variance))

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        # Create a color scheme: first normal and then fully transparent
        color = BasePlot._fig.get_elem("components")
        opacity = [0.2] * self.pca_._comps + [0] * (len(variance) - self.pca_._comps)

        fig.add_trace(
            go.Bar(
                x=variance,
                y=[f"pca{i}" for i in range(len(variance))],
                orientation="h",
                marker={
                    "color": [f"rgba({color[4:-1]}, {o})" for o in opacity],
                    "line": {"width": 2, "color": color},
                },
                hovertemplate="%{x}<extra></extra>",
                name=f"Variance retained: {variance[:self.pca_._comps].sum():.3f}",
                legendgroup="components",
                showlegend=BasePlot._fig.showlegend("components", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout({f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"}})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Explained variance ratio",
            ylim=(len(variance) - show_c - 0.5, len(variance) - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_components",
            filename=filename,
            display=display,
        )

    @crash
    def plot_correlation(
        self,
        columns: Segment | Sequence[Int | str] | DataFrame | None = None,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (800, 700),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a correlation matrix.

        Displays a heatmap showing the correlation between columns in
        the dataset. The colors red, blue and white stand for positive,
        negative, and no correlation respectively.

        Parameters
        ----------
        columns: segment, sequence, dataframe or None, default=None
            Columns to plot. If None, plot all columns in the dataset.
            Selected categorical columns are ignored.

        method: str, default="pearson"
            Method of correlation. Choose from: pearson, kendall or
            spearman.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Do nothing. Implemented for continuity of the API.

        figsize: tuple, default=(800, 700)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_qq
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.plot_correlation()
        ```

        """
        columns_c = self.branch._get_columns(columns, only_numerical=True)

        # Compute the correlation matrix
        corr = self.branch.dataset[columns_c].corr(method=method)

        # Generate a mask for the lower triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(
            x=(0, 0.87),
            coloraxis={
                "colorscale": "rdbu_r",
                "cmin": -1,
                "cmax": 1,
                "title": f"{method} correlation",
                "font_size": self.label_fontsize,
            },
        )

        fig.add_trace(
            go.Heatmap(
                z=corr.mask(mask),
                x=columns_c,
                y=columns_c,
                coloraxis=f"coloraxis{xaxis[1:]}",
                hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                hoverongaps=False,
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                "template": "plotly_white",
                f"yaxis{yaxis[1:]}_autorange": "reversed",
                f"xaxis{xaxis[1:]}_showgrid": False,
                f"yaxis{yaxis[1:]}_showgrid": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_correlation",
            filename=filename,
            display=display,
        )

    @crash
    def plot_distribution(
        self,
        columns: ColumnSelector = 0,
        distributions: str | Sequence[str] | None = "kde",
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot column distributions.

        - For numerical columns, plot the probability density
          distribution. Additionally, it's possible to plot any of
          `scipy.stats` distributions fitted to the column.
        - For categorical columns, plot the class distribution.
          Only one categorical column can be plotted at the same time.

        !!! tip
            Use atom's [distribution][atomclassifier-distribution]
            method to check which distribution fits the column best.

        Parameters
        ----------
        columns: int, str, slice or sequence, default=0
            Columns to plot. It's only possible to plot one categorical
            column. If more than one categorical column is selected,
            all categorical columns are ignored.

        distributions: str, sequence or None, default="kde"
            Distributions to fit. Only for numerical columns.

            - If None: No distribution is fit.
            - If "kde": Fit a [Gaussian kde distribution][kde].
            - Else: Name of a `scipy.stats` distribution.

        show: int or None, default=None
            Number of classes (ordered by number of occurrences) to
            show in the plot. If None, it shows all classes. Only for
            categorical columns.

        title: str, dict or None, default=None
            Title for the plot.

            - If None: No title is shown.
            - If str: Text for the title.
            - If dict: [title configuration][parameters].

        legend: str, dict or None, default="upper right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the plot's type.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_qq
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        import numpy as np
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add a categorical feature
        animals = ["cat", "dog", "bird", "lion", "zebra"]
        probabilities = [0.001, 0.1, 0.2, 0.3, 0.399]
        X["animals"] = np.random.choice(animals, size=len(X), p=probabilities)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.plot_distribution(columns=[0, 1])
        atom.plot_distribution(columns=0, distributions=["norm", "invgauss"])
        atom.plot_distribution(columns="animals")
        ```

        """
        columns_c = self.branch._get_columns(columns)
        num_columns = self.branch.dataset.select_dtypes(include="number").columns

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if len(columns_c) == 1 and columns_c[0] not in num_columns:
            series = self.branch.dataset[columns_c[0]].value_counts(ascending=True)
            show_c = self._get_show(show, len(series))

            color = BasePlot._fig.get_elem()
            fig.add_trace(
                go.Bar(
                    x=series,
                    y=series.index,
                    orientation="h",
                    marker={
                        "color": f"rgba({color[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": color},
                    },
                    hovertemplate="%{x}<extra></extra>",
                    name=f"{columns_c[0]}: {len(series)} classes",
                    showlegend=BasePlot._fig.showlegend("dist", legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Counts",
                ylim=(len(series) - show_c - 0.5, len(series) - 0.5),
                title=title,
                legend=legend,
                figsize=figsize or (900, 400 + show_c * 50),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

        else:
            for col in [c for c in columns_c if c in num_columns]:
                fig.add_trace(
                    go.Histogram(
                        x=self.branch.dataset[col],
                        histnorm="probability density",
                        marker={
                            "color": f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                            "line": {"width": 2, "color": BasePlot._fig.get_elem(col)},
                        },
                        nbinsx=40,
                        name="dist",
                        legendgroup=col,
                        legendgrouptitle={"text": col, "font_size": self.label_fontsize},
                        showlegend=BasePlot._fig.showlegend(f"{col}-dist", legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                x = np.linspace(
                    start=self.branch.dataset[col].min(),
                    stop=self.branch.dataset[col].max(),
                    num=200,
                )

                # Drop missing values for compatibility with scipy.stats
                values = replace_missing(self.branch.dataset[col], self.missing).dropna()
                values = values.to_numpy(dtype=float)

                if distributions is not None:
                    # Get a line for each distribution
                    for dist in lst(distributions):
                        if dist == "kde":
                            y = stats.gaussian_kde(values)(x)
                        else:
                            params = getattr(stats, dist).fit(values)
                            y = getattr(stats, dist).pdf(x, *params)

                        fig.add_trace(
                            self._draw_line(
                                x=x,
                                y=y,
                                parent=col,
                                child=dist,
                                legend=legend,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            )
                        )

            fig.update_layout({"barmode": "overlay"})

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Values",
                ylabel="Probability density",
                title=title,
                legend=legend,
                figsize=figsize or (900, 600),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

    @crash
    def plot_ngrams(
        self,
        ngram: Literal[1, 2, 3, 4, "word", "bigram", "trigram", "quadgram"] = "bigram",
        rows: RowSelector | None = "dataset",
        show: IntLargerZero | None = 10,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot n-gram frequencies.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised. If the documents are not tokenized, the words are
        separated by spaces.

        !!! tip
            Use atom's [tokenize][atomclassifier-tokenize] method to
            separate the words creating n-grams based on their frequency
            in the corpus.

        Parameters
        ----------
        ngram: str or int, default="bigram"
            Number of contiguous words to search for (size of n-gram).
            Choose from: word (1), bigram (2), trigram (3), quadgram (4).

        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Selection of rows][row-and-column-selection] in the corpus
            to include in the search.

        show: int or None, default=10
            Number of n-grams (ordered by number of occurrences) to
            show in the plot. If none, show all n-grams (up to 200).

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple or None, default=None
            Figure's size in pixels, format as (x, y). If None, it
            adapts the size to the number of n-grams shown.

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_wordcloud

        Examples
        --------
        ```pycon
        import numpy as np
        from atom import ATOMClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.textclean()
        atom.textnormalize()
        atom.plot_ngrams()
        ```

        """

        def get_text(column: Series) -> Series:
            """Get the complete corpus as sequence of tokens.

            Parameters
            ----------
            column: series
                Column containing the corpus.

            Returns
            -------
            series
                Corpus of tokens.

            """
            if isinstance(column.iloc[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        corpus = get_corpus(self.branch.X)
        rows_c = self.branch._get_rows(rows)
        show_c = self._get_show(show)

        if str(ngram) in ("1", "word"):
            ngram_c = "words"
            series = pd.Series(
                [word for row in get_text(rows_c[corpus]) for word in row]
            ).value_counts(ascending=True)
        else:
            if str(ngram) in ("2", "bigram"):
                ngram_c, finder = "bigrams", BigramCollocationFinder
            elif str(ngram) in ("3", "trigram"):
                ngram_c, finder = "trigrams", TrigramCollocationFinder
            elif str(ngram) in ("4", "quadgram"):
                ngram_c, finder = "quadgrams", QuadgramCollocationFinder

            ngram_fd = finder.from_documents(get_text(rows_c[corpus])).ngram_fd
            series = pd.Series(
                data=[x[1] for x in ngram_fd.items()],
                index=[" ".join(x[0]) for x in ngram_fd.items()],
            ).sort_values(ascending=True)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        fig.add_trace(
            go.Bar(
                x=(data := series[-self._get_show(show, len(series)):]),
                y=data.index,
                orientation="h",
                marker={
                    "color": f"rgba({BasePlot._fig.get_elem(ngram_c)[4:-1]}, 0.2)",
                    "line": {"width": 2, "color": BasePlot._fig.get_elem(ngram_c)},
                },
                hovertemplate="%{x}<extra></extra>",
                name=f"Total {ngram_c}: {len(series)}",
                legendgroup=ngram_c,
                showlegend=BasePlot._fig.showlegend(ngram_c, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Counts",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pca(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the explained variance ratio vs number of components.

        If the underlying estimator is [PCA][] (for dense datasets),
        all possible components are plotted. If the underlying estimator
        is [TruncatedSVD][] (for sparse datasets), it only shows the
        selected components. The star marks the number of components
        selected by the user. This plot is available only when feature
        selection was applied with strategy="pca".

        Parameters
        ----------
        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Do nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_components
        atom.plots:DataPlot.plot_rfecv

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("pca", n_features=5)
        atom.plot_pca()
        ```

        """
        if not hasattr(self, "pca_"):
            raise PermissionError(
                "The plot_components method is only available for instances "
                "that ran feature selection using the 'pca' strategy, "
                "e.g., atom.feature_selection(strategy='pca')."
            )

        # Create star symbol at selected number of components
        symbols = ["circle"] * self.pca_.n_features_in_
        symbols[self.pca_._comps - 1] = "star"
        sizes = [self.marker_size] * self.pca_.n_features_in_
        sizes[self.pca_._comps - 1] = self.marker_size * 1.5

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        fig.add_trace(
            go.Scatter(
                x=tuple(range(1, self.pca_.n_features_in_ + 1)),
                y=np.cumsum(self.pca_.explained_variance_ratio_),
                mode="lines+markers",
                line={"width": self.line_width, "color": BasePlot._fig.get_elem("pca")},
                marker={
                    "symbol": symbols,
                    "size": sizes,
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                    "opacity": 1,
                },
                hovertemplate="%{y}<extra></extra>",
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                "hovermode": "x",
                f"xaxis{xaxis[1:]}_showspikes": True,
                f"yaxis{yaxis[1:]}_showspikes": True,
            }
        )

        margin = self.pca_.n_features_in_ / 30
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            xlim=(1 - margin, self.pca_.n_features_in_ - 1 + margin),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pca",
            filename=filename,
            display=display,
        )

    @crash
    def plot_qq(
        self,
        columns: ColumnSelector = 0,
        distributions: str | Sequence[str] = "norm",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a quantile-quantile plot.

        Columns are distinguished by color and the distributions are
        distinguished by marker type. Missing values are ignored.

        Parameters
        ----------
        columns: int, str, slice or sequence, default=0
            Columns to plot. Selected categorical columns are ignored.

        distributions: str or sequence, default="norm"
            Names of the `scipy.stats` distributions to fit to the
            columns.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="lower right"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_relationships

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.plot_qq(columns=[5, 6])
        atom.plot_qq(columns=0, distributions=["norm", "invgauss", "triang"])
        ```

        """
        columns_c = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        percentiles = np.linspace(0, 100, 101)
        for col in columns_c:
            # Drop missing values for compatibility with scipy.stats
            values = replace_missing(self.branch.dataset[col], self.missing).dropna()
            values = values.to_numpy(dtype=float)

            for dist in lst(distributions):
                stat = getattr(stats, dist)
                params = stat.fit(values)
                samples = stat.rvs(*params, size=101, random_state=self.random_state)

                fig.add_trace(
                    self._draw_line(
                        x=np.percentile(samples, percentiles),
                        y=np.percentile(values, percentiles),
                        mode="markers",
                        parent=col,
                        child=dist,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        self._draw_straight_line(y="diagonal", xaxis=xaxis, yaxis=yaxis)

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Theoretical quantiles",
            ylabel="Observed quantiles",
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_qq",
            filename=filename,
            display=display,
        )

    @crash
    def plot_relationships(
        self,
        columns: Segment | Sequence[Int | str] | DataFrame = (0, 1, 2),
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 900),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot pairwise relationships in a dataset.

        Creates a grid of axes such that each numerical column appears
        once on the x-axes and once on the y-axes. The bottom triangle
        contains scatter plots (max 250 random samples), the diagonal
        plots contain column distributions, and the upper triangle
        contains contour histograms for all samples in the columns.

        Parameters
        ----------
        columns: segment, sequence or dataframe, default=(0, 1, 2)
            Columns to plot. Selected categorical columns are ignored.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Do nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 900)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_correlation
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_qq

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.plot_relationships(columns=[0, 4, 5])
        ```

        """
        columns_c = self.branch._get_columns(columns, only_numerical=True)

        # Use max 250 samples to not clutter the plot
        sample = lambda col: self.branch.dataset[col].sample(
            n=min(len(self.branch.dataset), 250), random_state=self.random_state
        )

        fig = self._get_figure()
        color = BasePlot._fig.get_elem()
        for i in range(len(columns_c) ** 2):
            x, y = i // len(columns_c), i % len(columns_c)

            # Calculate the distance between subplots
            offset = divide(0.0125, (len(columns_c) - 1))

            # Calculate the size of the subplot
            size = (1 - ((offset * 2) * (len(columns_c) - 1))) / len(columns_c)

            # Determine the position for the axes
            x_pos = y * (size + 2 * offset)
            y_pos = (len(columns_c) - x - 1) * (size + 2 * offset)

            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(x_pos, rnd(x_pos + size)),
                y=(y_pos, rnd(y_pos + size)),
                coloraxis={
                    "colorscale": PALETTE.get(color, "Blues"),
                    "cmin": 0,
                    "cmax": len(self.branch.dataset),
                    "showscale": False,
                },
            )

            if x == y:
                fig.add_trace(
                    go.Histogram(
                        x=self.branch.dataset[columns_c[x]],
                        marker={
                            "color": f"rgba({color[4:-1]}, 0.2)",
                            "line": {"width": 2, "color": color},
                        },
                        name=columns_c[x],
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif x > y:
                fig.add_trace(
                    go.Scatter(
                        x=sample(columns_c[y]),
                        y=sample(columns_c[x]),
                        mode="markers",
                        marker={"color": color},
                        hovertemplate="(%{x}, %{y})<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif y > x:
                fig.add_trace(
                    go.Histogram2dContour(
                        x=self.branch.dataset[columns_c[y]],
                        y=self.branch.dataset[columns_c[x]],
                        coloraxis=f"coloraxis{xaxis[1:]}",
                        hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

            if x < len(columns_c) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=columns_c[y] if x == len(columns_c) - 1 else None,
                ylabel=columns_c[x] if y == 0 else None,
            )

        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (900, 900),
            plotname="plot_relationships",
            filename=filename,
            display=display,
        )

    @crash
    def plot_rfecv(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot the rfecv results.

        Plot the scores obtained by the estimator fitted on every
        subset of the dataset. Only available when feature selection
        was applied with strategy="rfecv".

        Parameters
        ----------
        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_components
        atom.plots:DataPlot.plot_pca

        Examples
        --------
        ```pycon
        from atom import ATOMClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.feature_selection("rfecv", solver="Tree")
        atom.plot_rfecv()
        ```

        """
        if not hasattr(self, "rfecv_"):
            raise PermissionError(
                "The plot_rfecv method is only available for instances "
                "that ran feature selection using the 'rfecv' strategy, "
                "e.g., atom.feature_selection(strategy='rfecv')."
            )

        try:  # Define the y-label for the plot
            ylabel = self.rfecv_.get_params()["scoring"].name
        except AttributeError:
            ylabel = "accuracy" if is_classifier(self.rfecv_.estimator_) else "r2"

        x = range(self.rfecv_.min_features_to_select, self.rfecv_.n_features_in_ + 1)

        # Create star symbol at selected number of features
        sizes = [6] * len(x)
        sizes[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = 12
        symbols = ["circle"] * len(x)
        symbols[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = "star"

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        mean = self.rfecv_.cv_results_["mean_test_score"]
        std = self.rfecv_.cv_results_["std_test_score"]

        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=mean,
                mode="lines+markers",
                line={"width": self.line_width, "color": BasePlot._fig.get_elem("rfecv")},
                marker={
                    "symbol": symbols,
                    "size": sizes,
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                    "opacity": 1,
                },
                name=ylabel,
                legendgroup="rfecv",
                showlegend=BasePlot._fig.showlegend("rfecv", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        # Add error bands
        fig.add_traces(
            [
                go.Scatter(
                    x=tuple(x),
                    y=mean + std,
                    mode="lines",
                    line={"width": 1, "color": BasePlot._fig.get_elem("rfecv")},
                    hovertemplate="%{y}<extra>upper bound</extra>",
                    legendgroup="rfecv",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ),
                go.Scatter(
                    x=tuple(x),
                    y=mean - std,
                    mode="lines",
                    line={"width": 1, "color": BasePlot._fig.get_elem("rfecv")},
                    fill="tonexty",
                    fillcolor=f"rgba{BasePlot._fig.get_elem('rfecv')[3:-1]}, 0.2)",
                    hovertemplate="%{y}<extra>lower bound</extra>",
                    legendgroup="rfecv",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                ),
            ]
        )

        fig.update_layout({"hovermode": "x unified"})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=(min(x) - len(x) / 30, max(x) + len(x) / 30),
            ylim=(min(mean) - 3 * max(std), max(mean) + 3 * max(std)),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_rfecv",
            filename=filename,
            display=display,
        )

    @crash
    def plot_series(
        self,
        rows: str | Sequence[str] | dict[str, RowSelector] = ("train", "test"),
        columns: ColumnSelector | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plot a data series.

        This plot is specially useful to plot the time series for
        [forecast][time-series] tasks.

        Parameters
        ----------
        rows: str, sequence or dict, default=("train", "test")
            Selection of rows on which to calculate the metric.

            - If str: Name of the data set to plot.
            - If sequence: Names of the data sets to plot.
            - If dict: Names of the sets with corresponding
              [selection of rows][row-and-column-selection] as values.

        columns: int, str, segment, sequence, dataframe or None, default=None
            [Columns][row-and-column-selection] to plot. If None, all
            target columns are selected.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default="upper left"
            Legend for the plot. See the [user guide][parameters] for
            an extended description of the choices.

            - If None: No legend is shown.
            - If str: Location where to show the legend.
            - If dict: Legend configuration.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_distribution
        atom.plots:DataPlot.plot_relationships
        atom.plots:DataPlot.plot_qq

        Examples
        --------
        ```pycon
        from atom import ATOMForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        atom = ATOMForecaster(y, random_state=1)
        atom.plot_series()
        ```

        """
        if columns is None:
            columns_c = lst(self.target)
        else:
            columns_c = self.branch._get_columns(columns, include_target=True)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for col in columns_c:
            for child, ds in self._get_set(rows):
                fig.add_trace(
                    self._draw_line(
                        x=self._get_plot_index(y := self.branch._get_rows(ds)[col]),
                        y=y,
                        mode="lines+markers",
                        marker={
                            "size": self.marker_size,
                            "color": BasePlot._fig.get_elem(col),
                            "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                        },
                        parent=col,
                        child=child,
                        legend=legend,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel=self.branch.dataset.index.name or "index",
            ylabel="Values",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_series",
            filename=filename,
            display=display,
        )

    @crash
    def plot_wordcloud(
        self,
        rows: RowSelector = "dataset",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
        **kwargs,
    ) -> go.Figure | None:
        """Plot a wordcloud from the corpus.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised.

        Parameters
        ----------
        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Selection of rows][row-and-column-selection] in the corpus
            to include in the wordcloud.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Do nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str, Path or None, default=None
            Save the plot using this name. Use "auto" for automatic
            naming. The type of the file depends on the provided name
            (.html, .png, .pdf, etc...). If `filename` has no file type,
            the plot is saved as html. If None, the plot is not saved.

        display: bool or None, default=True
            Whether to render the plot. If None, it returns the figure.

        **kwargs
            Additional keyword arguments for the [Wordcloud][] object.

        Returns
        -------
        [go.Figure][] or None
            Plot object. Only returned if `display=None`.

        See Also
        --------
        atom.plots:DataPlot.plot_ngrams
        atom.plots:PredictionPlot.plot_pipeline

        Examples
        --------
        ```pycon
        import numpy as np
        from atom import ATOMClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        atom = ATOMClassifier(X, y, random_state=1)
        atom.textclean()
        atom.textnormalize()
        atom.plot_wordcloud()
        ```

        """

        def get_text(column):
            """Get the complete corpus as one long string."""
            if isinstance(column.iloc[0], str):
                return " ".join(column)
            else:
                return " ".join([" ".join(row) for row in column])

        check_dependency("wordcloud")
        from wordcloud import WordCloud

        corpus = get_corpus(self.branch.X)
        rows_c = self.branch._get_rows(rows)

        wordcloud = WordCloud(
            width=figsize[0],
            height=figsize[1],
            background_color=kwargs.pop("background_color", "white"),
            random_state=kwargs.pop("random_state", self.random_state),
            **kwargs,
        )

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        fig.add_trace(
            go.Image(
                z=wordcloud.generate(get_text(rows_c[corpus])),
                hoverinfo="skip",
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        fig.update_layout(
            {
                f"xaxis{xaxis[1:]}_showticklabels": False,
                f"yaxis{xaxis[1:]}_showticklabels": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_wordcloud",
            filename=filename,
            display=display,
        )
