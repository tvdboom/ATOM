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


def execute(src: str) -> str:
    """Get the code with output.

    Parameters
    ----------
    src: str
        Source code.

    Returns
    -------
    str
        Formatted code with output.

    """

    def draw(code: str) -> str:
        """Draw the code nicely formatted.

        Parameters
        ----------
        code: str
            Line of code to format.

        Returns
        -------
        str
            Formatted code.

        """
        if code == "":
            return code
        elif not code.startswith(" "):
            return f">>> {code}"
        else:
            return f"... {code}"

    ipy = InteractiveInterpreter()

    lines = src.split("\n")
    tree = {x.lineno: x for x in ast.parse(src).body}

    output = []
    for i, line in enumerate(lines, start=1):
        if node := tree.get(i):
            # Get complete code block
            block = lines[node.lineno - 1: node.end_lineno]

            if "# hide" not in line:
                output.extend([draw(code) for code in block])

            # Skip plotting since it's not rendered
            if ".plot_" in line:
                continue

            try:
                # Capture anything sent to stdout
                with StreamOut() as stream:
                    ipy.runsource("\n".join(block))

                    if text := stream.read():
                        output.append(f"\n{text}")

            except Exception as e:
                raise SuperFencesException from e

        else:
            output.append(draw(line))

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
    extension = md.preprocessors["fenced_code_block"].extension.superfences[0]

    return extension["formatter"](
        src=execute(src.strip()),
        class_name=css_class,
        language=language,
        md=md,
        options=options,
        **kwargs
    )
