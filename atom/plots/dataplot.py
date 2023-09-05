# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the DataPlot class.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from scipy import stats


from atom.plots.base import BasePlot
from atom.utils.constants import PALETTE
from atom.utils.types import INT, LEGEND, SEQUENCE, SERIES, SLICE
from atom.utils.utils import (
    check_dependency, composed, crash, divide, get_corpus, lst, rnd,
)


class DataPlot(BasePlot):
    """Data plots.

    Plots used for understanding and interpretation of the dataset.
    They are only accessible from atom since. The other runners should
    be used for model training only, not for data manipulation.

    """

    @crash
    def plot_correlation(
        self,
        columns: slice | SEQUENCE | None = None,
        method: str = "pearson",
        *,
        title: str | dict | None = None,
        legend: LEGEND | dict | None = None,
        figsize: tuple[INT, INT] = (800, 700),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot a correlation matrix.

        Displays a heatmap showing the correlation between columns in
        the dataset. The colors red, blue and white stand for positive,
        negative, and no correlation respectively.

        Parameters
        ----------
        columns: slice, sequence or None, default=None
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
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(800, 700)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
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
        columns = self.branch._get_columns(columns, only_numerical=True)
        if method.lower() not in ("pearson", "kendall", "spearman"):
            raise ValueError(
                f"Invalid value for the method parameter, got {method}. "
                "Choose from: pearson, kendall or spearman."
            )

        # Compute the correlation matrix
        corr = self.dataset[columns].corr(method=method.lower())

        # Generate a mask for the lower triangle
        # k=1 means keep outermost diagonal line
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(
            x=(0, 0.87),
            coloraxis=dict(
                colorscale="rdbu_r",
                cmin=-1,
                cmax=1,
                title=f"{method.lower()} correlation",
                font_size=self.label_fontsize,
            ),
        )

        fig.add_trace(
            go.Heatmap(
                z=corr.mask(mask),
                x=columns,
                y=columns,
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
        columns: SLICE = 0,
        distributions: str | SEQUENCE | None = None,
        show: INT | None = None,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "upper right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
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
            Columns to plot. I's only possible to plot one categorical
            column. If more than one categorical columns are selected,
            all categorical columns are ignored.

        distributions: str, sequence or None, default=None
            Names of the `scipy.stats` distributions to fit to the
            columns. If None, a [Gaussian kde distribution][kde] is
            showed. Only for numerical columns.

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

        filename: str or None, default=None
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
        columns = self.branch._get_columns(columns)
        cat_columns = list(self.dataset.select_dtypes(exclude="number").columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if len(columns) == 1 and columns[0] in cat_columns:
            series = self.dataset[columns[0]].value_counts(ascending=True)

            if show is None or show > len(series):
                show = len(series)
            elif show < 1:
                raise ValueError(
                    "Invalid value for the show parameter."
                    f"Value should be >0, got {show}."
                )

            color = BasePlot._fig.get_elem()
            fig.add_trace(
                go.Bar(
                    x=series,
                    y=series.index,
                    orientation="h",
                    marker=dict(
                        color=f"rgba({color[4:-1]}, 0.2)",
                        line=dict(width=2, color=color),
                    ),
                    hovertemplate="%{x}<extra></extra>",
                    name=f"{columns[0]}: {len(series)} classes",
                    showlegend=BasePlot._fig.showlegend("dist", legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            )

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Counts",
                ylim=(len(series) - show - 0.5, len(series) - 0.5),
                title=title,
                legend=legend,
                figsize=figsize or (900, 400 + show * 50),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

        else:
            for col in [c for c in columns if c not in cat_columns]:
                fig.add_trace(
                    go.Histogram(
                        x=self.dataset[col],
                        histnorm="probability density",
                        marker=dict(
                            color=f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                            line=dict(width=2, color=BasePlot._fig.get_elem(col)),
                        ),
                        nbinsx=40,
                        name="dist",
                        legendgroup=col,
                        legendgrouptitle=dict(text=col, font_size=self.label_fontsize),
                        showlegend=BasePlot._fig.showlegend(f"{col}-dist", legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

                x = np.linspace(self.dataset[col].min(), self.dataset[col].max(), 200)

                # Drop missing values for compatibility with scipy.stats
                missing = self.missing + [np.inf, -np.inf]
                values = self.dataset[col].replace(missing, np.NaN).dropna()

                if distributions:
                    # Get a line for each distribution
                    for j, dist in enumerate(lst(distributions)):
                        params = getattr(stats, dist).fit(values)

                        fig.add_trace(
                            self._draw_line(
                                x=x,
                                y=getattr(stats, dist).pdf(x, *params),
                                parent=col,
                                child=dist,
                                legend=legend,
                                xaxis=xaxis,
                                yaxis=yaxis,
                            )
                        )
                else:
                    # If no distributions specified, draw Gaussian kde
                    fig.add_trace(
                        self._draw_line(
                            x=x,
                            y=stats.gaussian_kde(values)(x),
                            parent=col,
                            child="kde",
                            legend=legend,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )
                    )

            fig.update_layout(dict(barmode="overlay"))

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
        ngram: INT | str = "bigram",
        index: SLICE | None = None,
        show: INT = 10,
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] | None = None,
        filename: str | None = None,
        display: bool | None = True,
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
            Choose from: words (1), bigrams (2), trigrams (3),
            quadgrams (4).

        index: int, str, slice, sequence or None, default=None
            Documents in the corpus to include in the search. If None,
            it selects all documents in the dataset.

        show: int, default=10
            Number of n-grams (ordered by number of occurrences) to
            show in the plot.

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

        filename: str or None, default=None
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

        def get_text(column: SERIES) -> SERIES:
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
            if isinstance(column.iat[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self.branch._get_rows(index, return_test=False)]

        if str(ngram).lower() in ("1", "word", "words"):
            ngram = "words"
            series = pd.Series(
                [word for row in get_text(rows[corpus]) for word in row]
            ).value_counts(ascending=True)
        else:
            if str(ngram).lower() in ("2", "bigram", "bigrams"):
                ngram, finder = "bigrams", BigramCollocationFinder
            elif str(ngram).lower() in ("3", "trigram", "trigrams"):
                ngram, finder = "trigrams", TrigramCollocationFinder
            elif str(ngram).lower() in ("4", "quadgram", "quadgrams"):
                ngram, finder = "quadgrams", QuadgramCollocationFinder
            else:
                raise ValueError(
                    f"Invalid value for the ngram parameter, got {ngram}. "
                    "Choose from: words, bigram, trigram, quadgram."
                )

            ngram_fd = finder.from_documents(get_text(rows[corpus])).ngram_fd
            series = pd.Series(
                data=[x[1] for x in ngram_fd.items()],
                index=[" ".join(x[0]) for x in ngram_fd.items()],
            ).sort_values(ascending=True)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        fig.add_trace(
            go.Bar(
                x=(data := series[-show:]),
                y=data.index,
                orientation="h",
                marker=dict(
                    color=f"rgba({BasePlot._fig.get_elem(ngram)[4:-1]}, 0.2)",
                    line=dict(width=2, color=BasePlot._fig.get_elem(ngram)),
                ),
                hovertemplate="%{x}<extra></extra>",
                name=f"Total {ngram}: {len(series)}",
                legendgroup=ngram,
                showlegend=BasePlot._fig.showlegend(ngram, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Counts",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show * 50),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @crash
    def plot_qq(
        self,
        columns: SLICE = 0,
        distributions: str | SEQUENCE = "norm",
        *,
        title: str | dict | None = None,
        legend: str | dict | None = "lower right",
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
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

        filename: str or None, default=None
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
        columns = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        percentiles = np.linspace(0, 100, 101)
        for col in columns:
            # Drop missing values for compatibility with scipy.stats
            missing = self.missing + [np.inf, -np.inf]
            values = self.dataset[col].replace(missing, np.NaN).dropna()

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
        columns: slice | SEQUENCE = (0, 1, 2),
        *,
        title: str | dict | None = None,
        legend: LEGEND | dict | None = None,
        figsize: tuple[INT, INT] = (900, 900),
        filename: str | None = None,
        display: bool | None = True,
    ) -> go.Figure | None:
        """Plot pairwise relationships in a dataset.

        Creates a grid of axes such that each numerical column appears
        once on the x-axes and once on the y-axes. The bottom triangle
        contains scatter plots (max 250 random samples), the diagonal
        plots contain column distributions, and the upper triangle
        contains contour histograms for all samples in the columns.

        Parameters
        ----------
        columns: slice or sequence, default=(0, 1, 2)
            Columns to plot. Selected categorical columns are ignored.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 900)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
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
        columns = self.branch._get_columns(columns, only_numerical=True)

        # Use max 250 samples to not clutter the plot
        sample = lambda col: self.dataset[col].sample(
            n=min(len(self.dataset), 250), random_state=self.random_state
        )

        fig = self._get_figure()
        color = BasePlot._fig.get_elem()
        for i in range(len(columns)**2):
            x, y = i // len(columns), i % len(columns)

            # Calculate the distance between subplots
            offset = divide(0.0125, (len(columns) - 1))

            # Calculate the size of the subplot
            size = (1 - ((offset * 2) * (len(columns) - 1))) / len(columns)

            # Determine the position for the axes
            x_pos = y * (size + 2 * offset)
            y_pos = (len(columns) - x - 1) * (size + 2 * offset)

            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(x_pos, rnd(x_pos + size)),
                y=(y_pos, rnd(y_pos + size)),
                coloraxis=dict(
                    colorscale=PALETTE.get(color, "Blues"),
                    cmin=0,
                    cmax=len(self.dataset),
                    showscale=False,
                )
            )

            if x == y:
                fig.add_trace(
                    go.Histogram(
                        x=self.dataset[columns[x]],
                        marker=dict(
                            color=f"rgba({color[4:-1]}, 0.2)",
                            line=dict(width=2, color=color),
                        ),
                        name=columns[x],
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif x > y:
                fig.add_trace(
                    go.Scatter(
                        x=sample(columns[y]),
                        y=sample(columns[x]),
                        mode="markers",
                        marker=dict(color=color),
                        hovertemplate="(%{x}, %{y})<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )
            elif y > x:
                fig.add_trace(
                    go.Histogram2dContour(
                        x=self.dataset[columns[y]],
                        y=self.dataset[columns[x]],
                        coloraxis=f"coloraxis{xaxis[1:]}",
                        hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )
                )

            if x < len(columns) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=columns[y] if x == len(columns) - 1 else None,
                ylabel=columns[x] if y == 0 else None,
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
    def plot_wordcloud(
        self,
        index: SLICE | None = None,
        *,
        title: str | dict | None = None,
        legend: LEGEND | dict | None = None,
        figsize: tuple[INT, INT] = (900, 600),
        filename: str | None = None,
        display: bool | None = True,
        **kwargs,
    ) -> go.Figure | None:
        """Plot a wordcloud from the corpus.

        The text for the plot is extracted from the column named
        `corpus`. If there is no column with that name, an exception
        is raised.

        Parameters
        ----------
        index: int, str, slice, sequence or None, default=None
            Documents in the corpus to include in the wordcloud. If
            None, it selects all documents in the dataset.

        title: str, dict or None, default=None
            Title for the plot.

            - If None, no title is shown.
            - If str, text for the title.
            - If dict, [title configuration][parameters].

        legend: str, dict or None, default=None
            Does nothing. Implemented for continuity of the API.

        figsize: tuple, default=(900, 600)
            Figure's size in pixels, format as (x, y).

        filename: str or None, default=None
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
            if isinstance(column.iat[0], str):
                return " ".join(column)
            else:
                return " ".join([" ".join(row) for row in column])

        check_dependency("wordcloud")
        from wordcloud import WordCloud

        corpus = get_corpus(self.X)
        rows = self.dataset.loc[self.branch._get_rows(index, return_test=False)]

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
                z=wordcloud.generate(get_text(rows[corpus])),
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
