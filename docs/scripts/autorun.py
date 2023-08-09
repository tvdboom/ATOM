# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the automatic example rendering.

"""

from __future__ import annotations

import ast
import sys
from code import InteractiveInterpreter
from io import StringIO

from markdown import Markdown
from pymdownx.superfences import SuperFencesException


class StreamOut:
    """Override stdout to fetch the code's output."""

    def __init__(self):
        self.old = sys.stdout
        self.stdout = StringIO()
        sys.stdout = self.stdout

    def read(self):
        """Read the stringIO buffer."""
        value = ""
        if self.stdout is not None:
            self.stdout.flush()
            value = self.stdout.getvalue()
            self.stdout = StringIO()
            sys.stdout = self.stdout

        return value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old
        self.old = None
        self.stdout = None


def get_code(src: str) -> str:
    """Get the code, nicely formatted.

    Adds or ... in front of every line.

    Parameters
    ----------
    src: str
        Source code.

    Returns
    -------
    str
        Formatted source.

    """
    code = []
    for i, line in enumerate(src.split("\n")):
        if "# hide" in line:
            continue
        elif line == "":
            code.append("")
        elif not line.startswith(" "):
            code.append(f">>> {line}")
        else:
            code.append(f"... {line}")

    return "\n".join(code)


def get_output(src: str) -> str:
    """Get the code's output.

    Parameters
    ----------
    src: str
        Source code.

    Returns
    -------
    str
        Code output.

    """
    output = []
    ipy = InteractiveInterpreter()

    tree = ast.parse(src)
    lines = src.split("\n")

    for node in tree.body:
        payload = "\n".join(lines[node.lineno - 1: node.end_lineno])

        # Skip plotting since it's not rendered
        if ".plot_" in payload:
            continue

        try:
            # Capture anything sent to stdout
            with StreamOut() as stream:
                ipy.runsource(payload)

                if text := stream.read():
                    output.append(text)

        except Exception as e:
            raise SuperFencesException from e

    return "\n".join(output)


def formatter(
    src: str,
    language: str,
    css_class: str,
    options: dict | None,
    md: Markdown,
    **kwargs,
) -> str:
    """Formatter wrapper.

    Parameters
    ----------
    src: str
        Source code.

    language: str
        Programming language used in `src`.

    css_class: str
        Name of the css class to add to the block.

    options: dict or None
        Additional options for the formatter.

    md: markdown.Markdown
        Markdown object (in case you want access to metadata).

    **kwargs
        Additional keyword arguments for the formatter.

    Returns
    -------
    str
        Source formatted to HTML.

    """
    src = src.strip()
    extension = md.preprocessors["fenced_code_block"].extension.superfences[0]

    return extension["formatter"](
        src=f"{get_code(src)}\n\n{get_output(src)}",
        class_name=css_class,
        language="pycon",
        md=md,
        options=options,
        **kwargs
    )
