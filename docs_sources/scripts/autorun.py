"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the automatic example rendering.

"""

import ast
import os
import shutil
import sys
from base64 import b64encode
from code import InteractiveInterpreter
from io import StringIO
from uuid import uuid4

import matplotlib as mpl
import plotly.io as pio
from markdown import Markdown
from pymdownx.superfences import SuperFencesException


# Avoid plot rendering
mpl.use("Agg")
pio.renderers.default = "pdf"

# Directory in which to store all plots from the examples
shutil.rmtree(DIR_EXAMPLES := "docs_sources/img/examples/", ignore_errors=True)
os.mkdir(DIR_EXAMPLES)


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


def execute(src: str) -> tuple[list[str], list[str]]:
    """Get the code with output.

    Parameters
    ----------
    src: str
        Source code.

    Returns
    -------
    list of str
        Blocks of formatted code with output.

    list of str
        Figures corresponding to every code block (can be empty).

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
        elif not code.startswith((" ", ")", "]", "}")):
            return f">>> {code}"
        else:
            return f"... {code}"

    def latest_file() -> str | None:
        """Get the most recent file from the plots directory.

        Returns
        -------
        str or None
            Name of the file. Returns None if the dir is empty.

        """
        if files := [os.path.join(DIR_EXAMPLES, f) for f in os.listdir(DIR_EXAMPLES)]:
            return os.path.basename(max(files, key=os.path.getmtime))

    ipy = InteractiveInterpreter()

    lines = src.split("\n")
    tree = {x.lineno: x for x in ast.parse(src).body}

    end_line = 0
    output, figures = [[]], []
    for i, line in enumerate(lines, start=1):
        if node := tree.get(i):
            end_line = node.end_lineno

            # Get complete code block
            block = lines[node.lineno - 1 : end_line]

            if "# hide" not in line:
                output[-1].extend([draw(code) for code in block])

            # Add filename parameter to plot call to save the figure
            if ".plot_" in line or ".canvas(" in line:
                f, arguments = block[0].split("(", 1)
                if arguments.startswith(")"):
                    # There are no other arguments
                    block[0] = f'{f}(filename="{DIR_EXAMPLES}{uuid4()}")'
                else:
                    # Attach filename after other arguments
                    args, close = arguments.rsplit(")", 1)
                    block[0] = f'{f}({args}, filename="{DIR_EXAMPLES}{uuid4()}"){close}'

            # Capture anything sent to stdout
            with StreamOut() as stream:
                # Add \n at end to exit contextmanagers
                ipy.runsource("\n".join(block) + "\n")

                if text := stream.read():
                    # Omit plot's output
                    if not text.startswith("{'application/pdf'"):
                        output[-1].append(f"\n{text}")

            if ".plot_" in line or ".canvas(" in line:
                if end_line < len(lines):
                    output.append([])  # Add new code block

                if (f := latest_file()).endswith(".html"):
                    with open(f"{DIR_EXAMPLES}{f}", encoding="utf-8") as file:
                        figures.append(file.read())
                else:
                    with open(f"{DIR_EXAMPLES}{f}", "rb") as file:
                        img = b64encode(file.read()).decode("utf-8")
                    figures.append(
                        f"<img src='data:image/png;base64,{img}' alt='{f}' draggable='false'>"
                    )

        elif i > end_line:
            output[-1].append(draw(line))

    return output, figures


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
        Markdown object.

    **kwargs
        Additional keyword arguments for the formatter.

    Returns
    -------
    str
        Source formatted to HTML.

    """

    def to_html(code: list[str]) -> str:
        """Convert a code block to html.

        Parameters
        ----------
        code: list of str
            List of commands.

        Returns
        -------
        str
            Clean representation of the code.

        """
        return md.preprocessors["fenced_code_block"].extension.superfences[0]["formatter"](
            src="\n".join(code),
            class_name=css_class,
            language=language,
            md=md,
            options=options,
            **kwargs,
        )

    # First line of markdown page
    print(md.lines[0])

    try:
        output, figures = execute(src.strip())

        render = []
        for i in range(max(len(output), len(figures))):
            source = ""
            if i < len(output):
                source += to_html(output[i])
            if i < len(figures):
                source += figures[i]

            render.append(source)

    except Exception as ex:
        raise SuperFencesException(f"Exception raised running code:\n{src}") from ex

    return "<br>".join(render)
