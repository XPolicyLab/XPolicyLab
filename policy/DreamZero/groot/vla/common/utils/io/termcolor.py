# coding: utf-8
# Copyright (c) 2008-2011 Volvox Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Original Author: Konstantin Lepa <konstantin.lepa@gmail.com>
# Updated by Jim Fan

"""ANSII Color formatting for output in terminal."""
import io
import os
from typing import List, Optional, Union

__ALL__ = ["color_text", "cprint"]

STYLES = dict(
    list(
        zip(
            ["bold", "dark", "", "underline", "blink", "", "reverse", "concealed"],
            list(range(1, 9)),
        )
    )
)
del STYLES[""]


HIGHLIGHTS = dict(
    list(
        zip(
            ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"],
            list(range(40, 48)),
        )
    )
)


COLORS = dict(
    list(
        zip(
            ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"],
            list(range(30, 38)),
        )
    )
)


def _strip_bg_prefix(color):
    "on_red -> red"
    if color.startswith("on_"):
        return color[len("on_") :]
    else:
        return color


RESET = "\033[0m"


def color_text(
    text,
    color: Optional[str] = None,
    bg_color: Optional[str] = None,
    styles: Optional[Union[str, List[str]]] = None,
):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv("ANSI_COLORS_DISABLED") is None:
        fmt_str = "\033[%dm%s"
        if color is not None:
            text = fmt_str % (COLORS[color], text)

        if bg_color is not None:
            bg_color = _strip_bg_prefix(bg_color)
            text = fmt_str % (HIGHLIGHTS[bg_color], text)

        if styles is not None:
            if isinstance(styles, str):
                styles = [styles]
            for style in styles:
                text = fmt_str % (STYLES[style], text)

        text += RESET
    return text


def cprint(
    *args,
    color: Optional[str] = None,
    bg_color: Optional[str] = None,
    styles: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """Print colorize text.

    It accepts arguments of print function.
    """
    sstream = io.StringIO()
    print(*args, sep=kwargs.pop("sep", None), end="", file=sstream)
    text = sstream.getvalue()
    print((color_text(text, color, bg_color, styles)), **kwargs)


if __name__ == "__main__":
    print("Current terminal type: %s" % os.getenv("TERM"))
    print("Test basic colors:")
    cprint("Grey color", color="grey")
    cprint("Red color", color="red")
    cprint("Green color", color="green")
    cprint("Yellow color", color="yellow")
    cprint("Blue color", color="blue")
    cprint("Magenta color", color="magenta")
    cprint("Cyan color", color="cyan")
    cprint("White color", color="white")
    print(("-" * 78))

    print("Test highlights:")
    cprint("On grey color", bg_color="on_grey")
    cprint("On red color", bg_color="on_red")
    cprint("On green color", bg_color="on_green")
    cprint("On yellow color", bg_color="on_yellow")
    cprint("On blue color", bg_color="on_blue")
    cprint("On magenta color", bg_color="on_magenta")
    cprint("On cyan color", bg_color="on_cyan")
    cprint("On white color", color="grey", bg_color="on_white")
    print("-" * 78)

    print("Test attributes:")
    cprint("Bold grey color", color="grey", styles="bold")
    cprint("Dark red color", color="red", styles=["dark"])
    cprint("Underline green color", color="green", styles=["underline"])
    cprint("Blink yellow color", color="yellow", styles=["blink"])
    cprint("Reversed blue color", color="blue", styles=["reverse"])
    cprint("Concealed Magenta color", color="magenta", styles=["concealed"])
    cprint(
        "Bold underline reverse cyan color",
        color="cyan",
        styles=["bold", "underline", "reverse"],
    )
    cprint(
        "Dark blink concealed white color",
        color="white",
        styles=["dark", "blink", "concealed"],
    )
    print(("-" * 78))

    print("Test mixing:")
    cprint(
        "Underline red on grey color",
        color="red",
        bg_color="on_grey",
        styles="underline",
    )
    cprint(
        "Reversed green on red color",
        color="green",
        bg_color="on_red",
        styles="reverse",
    )
