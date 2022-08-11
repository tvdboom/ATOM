# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the documentation rendering.

"""

from __future__ import annotations

import importlib
import inspect
from inspect import Parameter, getmembers, getsourcelines, signature
from typing import Callable, List, Optional

import regex as re
import yaml
from markdown import markdown


# Variables ======================================================== >>

# Mapping of keywords to urls
# Usage in docs: [anchor][key] -> [anchor][value]
CUSTOM_URLS = dict(
    # FeatureSelector
    selectkbest="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html",
    pca="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
    sfm="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html",
    sfs="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html",
    rfe="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html",
    rfecv="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html",
    pso="https://jaswinder9051998.github.io/zoofs/Particle%20Swarm%20Optimization%20Class/",
    hho="https://jaswinder9051998.github.io/zoofs/Harris%20Hawk%20Optimization/",
    gwo="https://jaswinder9051998.github.io/zoofs/Grey%20Wolf%20Optimization%20Class/",
    dfo="https://jaswinder9051998.github.io/zoofs/Dragon%20Fly%20Optimization%20Class/",
    go="https://jaswinder9051998.github.io/zoofs/Genetic%20Optimization%20Class/",
    pearson="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
    f_classif="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html",
    f_regression="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html",
    mutual_info_classif="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html",
    mutual_info_regression="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html",
    chi2="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html",
    # Ensembles
    votingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html",
    votingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html",
    stackingclassifier="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html",
    stackingregressor="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html",
)


# Functions ======================================================== >>

def render(markdown: str, **kwargs) -> str:
    """Render the markdown page.

    This function is the landing point for the mkdocs-simple-hooks
    plugin, called in mkdocs.yml.

    Parameters
    ----------
    markdown: str
        Markdown source text of page.

    **kwargs
        Additional keyword arguments of the hook.
            - page: Mkdocs Page instance.
            - config: Global configuration object.
            - files: Global files collection.

    Returns
    -------
    str
        Modified markdown/html source text of page.

    """
    while match := re.search("(:: )(\w.*?)(?=::|\n\n|\Z)", markdown, re.S):
        command = yaml.safe_load(match.group(2))

        # Commands should always be dicts with the configuration as a list in values
        if isinstance(command, str):
            if ":" in command:
                obj = AutoDocs.get_obj(command)
                markdown = markdown[:match.start()] + markdown[match.end():]
                continue
            else:
                command = {command: None}  # Has no options specified

        if "signature" in command:
            text = obj.get_signature()
        elif "description" in command:
            text = obj.get_summary() + "\n\n" + obj.get_description()
        elif "table" in command:
            text = obj.get_table(command["table"])
        elif "see also" in command:
            text = obj.get_see_also()
        elif "notes" in command:
            text = obj.get_block("Notes")
        elif "references" in command:
            text = obj.get_block("References")
        elif "examples" in command:
            text = obj.get_block("Examples")
        elif "methods" in command:
            text = obj.get_methods(command["methods"] or {})
        else:
            text = ""

        markdown = markdown[:match.start()] + text + markdown[match.end():]

    return custom_autorefs(markdown)


def custom_autorefs(markdown):
    """Custom handling of autorefs links.

    ATOM's documentation accepts some custom formatting for autorefs
    links in order to make the documentation cleaner and easier to
    write. The custom transformations are:
        - Replace keywords with full url (registered in CUSTOM_URLS).
        - Replace spaces with dashes.
        - Convert all links to lower case.

    """
    result, start = "", 0
    for match in re.finditer("\[([ \w_-]*?)\]\[([ \w_-]*?)\]", markdown):
        anchor = match.group(1)
        link = match.group(2)

        text = match.group()
        if not link:
            # Only adapt when has form [anchor][]
            link = anchor.replace(' ', '-').lower()
            text = f"[{anchor}][{link}]"
        if link in CUSTOM_URLS:
            # Replace keyword with custom url
            text = f"[{anchor}]({CUSTOM_URLS[link]})"

        result += markdown[start:match.start()] + text
        start = match.end()

    return result + markdown[start:]


# Classes ========================================================== >>

class AutoDocs:
    """Parses an object to documentation in markdown/html.

    The docstring should follow the numpydoc style[^1]. Blocks should
    start with `::`. The following blocks are accepted:
        - summary (first line of docstring, required)
        - description (detailed explanation, can contain admonitions)
        - parameters
        - attributes
        - returns
        - raises
        - see also
        - notes
        - references
        - examples

    Parameters
    ----------
    obj: callable
        Class, method or function to parse.

    method: callable or None
        Method of the obj to parse.

    References
    ----------
    [1] https://numpydoc.readthedocs.io/en/latest/format.html

    """

    blocks = (
        "Parameters",
        "Attributes",
        "Returns",
        "Raises",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "\Z",
    )

    def __init__(self, obj: Callable, method: Optional[Callable] = None):
        if method:
            self.obj = getattr(obj, method)
            self.method = method
            self._parent_anchor = obj.__name__.lower() + "-"
        else:
            self.obj = obj
            self.method = method
            self._parent_anchor = ""

        lines = self.obj.__doc__.splitlines()
        if len(lines) > 1:
            doc = self.obj.__doc__.splitlines()[2:]
            row = next(filter(lambda x: x != "", doc))  # Select first non-empty line
            indent = len(row) - len(row.lstrip())  # Measure block indentation
            self.doc = "\n".join([line[indent:] for line in doc])
        else:
            self.doc = ""

    @staticmethod
    def get_obj(command: str) -> AutoDocs:
        """Get an AutoDocs object from a string.

        The provided string must be of the form module:object or
        module:object.method.

        Parameters
        ----------
        command: str
            Line with the module and object.

        Returns
        -------
        Autodocs
            New instance of the class.

        """
        module, name = command.split(":")
        if "." in name:
            name, method = name.split(".")
            cls = getattr(importlib.import_module(module), name)
            return AutoDocs(getattr(cls, method))
        else:
            return AutoDocs(getattr(importlib.import_module(module), name))

    def get_signature(self) -> str:
        """Return the object's signature."""
        # Assign object type
        params = signature(self.obj).parameters
        if inspect.isclass(self.obj):
            obj_type = "class"
        elif "self" in params or "cls" in params:
            obj_type = "method"
        else:
            obj_type = "function"

        # Get signature without self, cls and type hints
        sign = []
        for k, v in params.items():
            if k not in ("self", "cls"):
                if v.default == Parameter.empty:
                    if '**' in str(v):
                        sign.append(f"**{k}")  # Add ** to kwargs
                    elif '*' in str(v):
                        sign.append(f"*{k}")  # Add * to args
                    else:
                        sign.append(k)
                else:
                    if isinstance(v.default, str):
                        sign.append(f'{k}="{v.default}"')
                    else:
                        sign.append(f"{k}={v.default}")

        sign = f"({', '.join(sign)})"

        f = self.obj.__module__.replace('.', '/')  # Module and filename sep by /
        if "atom" in self.obj.__module__:
            url = f"https://github.com/tvdboom/ATOM/blob/master/{f}.py"
        elif "sklearn" in self.obj.__module__:
            url = f"https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/{f}.py"
        else:
            url = ""

        anchor = f"<a id='{self._parent_anchor}{self.obj.__name__}'></a>"
        module = self.obj.__module__ + '.' if obj_type != "method" else ""
        obj_type = f"<em>{obj_type}</em>"
        name = f"<strong style='color:#008AB8'>{self.obj.__name__}</strong>"
        if url:
            line = getsourcelines(self.obj)[1]
            url = f"<span style='float:right'><a href={url}#L{line}>[source]</a></span>"

        return f"{anchor}<div class='sign'>{obj_type} {module}{name}{sign}{url}</div>"

    def get_summary(self) -> str:
        """Return the object's summary."""
        if self.obj.__doc__.splitlines()[0] == "":
            return self.obj.__doc__.splitlines()[1].lstrip()
        else:
            return self.obj.__doc__.splitlines()[0].lstrip()

    def get_description(self) -> str:
        """Return the object's description."""
        match = re.match(f".*?(?={'|'.join(self.blocks)})", self.doc, re.S)
        return match.group() if match else ""

    def get_see_also(self) -> str:
        """Return the object's See Also block.

        The block is rendered as an info admonition.

        Returns
        -------
        str
            Formatted block.

        """
        block = '!!! info "See Also"'
        for line in self.get_block("See Also").splitlines():
            if line:
                cls = self.get_obj(line)
                summary = f"<div style='margin: -1em 0 0 1.2em'>{cls.get_summary()}</div>"

                # If it's a class, refer to the page, else to the anchor
                if cls._parent_anchor:
                    link = f"{cls._parent_anchor}-{cls.obj.__name__}"
                else:
                    link = ""

                block += f"\n    [{cls.obj.__name__}][{link}]<br>    {summary}\n"

        return block

    def get_block(self, block: str) -> str:
        """Return a block from the docstring.

        Parameters
        ----------
        block: str
            Name of the block to retrieve.

        Returns
        -------
        str
            Block in docstring.

        """
        pattern = f"(?<={block}\n{'-' * len(block)}).*?(?={'|'.join(self.blocks)})"
        match = re.search(pattern, self.doc, re.S)
        return match.group() if match else ""

    def get_table(self, blocks: List[str]) -> str:
        """Return a table from a block.

        Parameters
        ----------
        blocks: list of str
            Blocks to create the table from.

        Returns
        -------
        str
            Table in html format.

        """
        table = ""
        for block in map(str.capitalize, blocks):
            match = self.get_block(block)

            if match:
                # Create td of title
                table += f"<tr><td class='td_title'><strong>{block}</strong></td>"

                # Create td of body
                table += "<td class='td_params'>"

                # Headers start with letter, * or [ after new line
                for header in re.findall("^[\[*\w].*?$", match, re.M):
                    # Get the body corresponding to the header
                    pattern = f"(?<={re.escape(header)}\n).*?(?=\n\w|\n\*|\n\[|\Z)"
                    body = re.search(pattern, match, re.S | re.M).group()

                    # Use literal * for args and kwargs
                    header = header.replace("*", "\*")

                    text = body
                    text = f"<div markdown style='margin: 0 0 1em 1.2em'>{text}</div>"
                    table += f"<strong>{header}</strong><br>{text}"

                table += "</td></tr>"

        if table:
            table = "<table markdown class='table_params'>" + table + "</table>"

        return table

    def get_methods(self, config: dict) -> str:
        """Return an overview of the methods and their blocks.

        Parameters
        ----------
        config: dict
            Options to configure. Choose from:
                - toc_only: Whether to display only the toc.
                - url: Page to link the toc to. None for current.
                - include: Members to include.
                - exclude: Members to exclude.

        Returns
        -------
        str
            Toc and blocks for all selected methods.

        """
        if config.get("include"):
            methods = config["include"]
        else:
            methods = [
                m for m, _ in getmembers(self.obj, predicate=inspect.isfunction)
                if not m.startswith("_") and m not in config.get("exclude", [])
            ]

        # Create toc
        toc = "<table markdown style='font-size: 0.9em'>"
        for method in methods:
            func = AutoDocs(self.obj, method=method)

            name = f"[{method}][{func._parent_anchor}{method}]"
            summary = func.get_summary()
            toc += f"<tr><td>{name}</td><td>{summary}</td></tr>"

        toc += "</table>"

        # Create methods
        blocks = ""
        if not config.get("toc_only"):
            for method in methods:
                func = AutoDocs(self.obj, method=method)

                blocks += "<br>" + func.get_signature()
                blocks += func.get_summary() + "\n"
                if func.obj.__module__.startswith("atom"):
                    blocks += "\n" + func.get_description()
                blocks += func.get_table(["Parameters", "Returns"]) + "<br>"

        return toc + blocks
