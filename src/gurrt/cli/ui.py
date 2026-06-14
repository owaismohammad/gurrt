from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
)
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.lexers import Lexer

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE — edit here to retheme the entire CLI
# ══════════════════════════════════════════════════════════════════════════════

# Rich color names — used in theme tokens, panel borders, rules, progress bars
C_ACCENT   = "yellow1"       # primary accent: headings, borders, prompt arrow
C_SUCCESS  = "bright_green"  # success messages and borders
C_ERROR    = "bright_red"    # error messages and borders
C_WARNING  = "orange1"       # warnings (orange — distinct from yellow accent)
C_DIM      = "dim"           # muted / secondary text

# Panel / table / rule border colors
BORDER_PRIMARY = "#0000ff"    # informational panels — blue
BORDER_SUCCESS = C_SUCCESS    # success panels
BORDER_ERROR   = C_ERROR      # error panels
BORDER_WARNING = C_WARNING    # warning panels

# Table header style
STYLE_HEADER = f"bold {C_ACCENT}"

# ── Prompt-toolkit REPL colors ────────────────────────────────────────────────
# ANSI names or #rrggbb hex — controls the interactive prompt and slash-command
# completion popup that appears when the user types "/"
_PT_PROMPT_COLOR  = "#ffff00"  # "gurrt" name in the prompt line — bright yellow
_PT_ARROW_COLOR   = "#ffff00"  # "❯" arrow — matches prompt name
_PT_DOT_ON        = "ansigreen" # ● indicator when a video is indexed
_PT_DOT_OFF       = "ansigray"  # ○ indicator when nothing is indexed
_PT_POPUP_BG      = "#00003a"   # completion popup background — dark blue
_PT_POPUP_SEL_BG  = "#0000aa"   # selected entry background — classic blue
_PT_POPUP_FG      = "#c8c800"   # unselected entry foreground — muted yellow
_PT_POPUP_SEL_FG  = "#ffff00"   # selected entry foreground — bright yellow
_PT_META_FG       = "#444488"   # meta / description text (unselected) — muted blue
_PT_META_SEL_FG   = "#8888ff"   # meta / description text (selected) — light blue
_PT_SCROLLBAR_BG  = "#00003a"   # scrollbar track — dark blue
_PT_SCROLLBAR_BTN = "#0000aa"   # scrollbar button — classic blue
_PT_CMD_COLOR     = "ansiyellow" # /command text in the input line
_PT_ARGS_COLOR    = "ansiwhite"  # arguments typed after the command

# ══════════════════════════════════════════════════════════════════════════════

_theme = Theme({
    "primary": f"bold {C_ACCENT}",
    "success": C_SUCCESS,
    "error":   f"bold {C_ERROR}",
    "warning": C_WARNING,
    "info":    C_DIM,
})

console = Console(theme=_theme)

pt_style = PromptStyle.from_dict({
    "prompt-name":    f"bold {_PT_PROMPT_COLOR}",
    "prompt-dot-on":  f"bold {_PT_DOT_ON}",
    "prompt-dot-off": _PT_DOT_OFF,
    "prompt-arrow":   f"bold {_PT_ARROW_COLOR}",
    # completion popup
    "completion-menu.completion":              f"bg:{_PT_POPUP_BG} fg:{_PT_POPUP_FG}",
    "completion-menu.completion.current":      f"bg:{_PT_POPUP_SEL_BG} fg:{_PT_POPUP_SEL_FG} bold",
    "completion-menu.meta.completion":         f"bg:{_PT_POPUP_BG} fg:{_PT_META_FG}",
    "completion-menu.meta.completion.current": f"bg:{_PT_POPUP_SEL_BG} fg:{_PT_META_SEL_FG}",
    "scrollbar.background": f"bg:{_PT_SCROLLBAR_BG}",
    "scrollbar.button":     f"bg:{_PT_SCROLLBAR_BTN}",
    # input text highlighting (via SlashCommandLexer)
    "slash-cmd":  f"bold {_PT_CMD_COLOR}",
    "slash-args": _PT_ARGS_COLOR,
})


class SlashCommandLexer(Lexer):
    """Highlights /command text in the REPL input line as the user types."""

    def lex_document(self, document):
        lines = document.lines

        def get_line(lineno: int):
            line = lines[lineno]
            if not line.startswith("/"):
                return [("", line)]
            idx = line.find(" ")
            if idx == -1:
                return [("class:slash-cmd", line)]
            return [
                ("class:slash-cmd",  line[:idx]),
                ("class:slash-args", line[idx:]),
            ]

        return get_line


def get_prompt_tokens(indexed: bool) -> FormattedText:
    """Return the styled REPL prompt tokens for prompt_toolkit."""
    return FormattedText([
        ("", "\n"),
        ("class:prompt-name", "gUrrT"),
        ("", " "),
        ("class:prompt-dot-on" if indexed else "class:prompt-dot-off", "●" if indexed else "○"),
        ("", " "),
        ("class:prompt-arrow", "❯"),
        ("", " "),
    ])


_BANNER = r"""
 ██████╗ ██╗   ██╗██████╗ ██████╗ ████████╗
██╔════╝ ██║   ██║██╔══██╗██╔══██╗╚══██╔══╝
██║  ███╗██║   ██║██████╔╝██████╔╝   ██║
██║   ██║╚██╗ ██╔╝██╔══██╗██╔══██╗   ██║
╚██████╔╝ ╚████╔╝ ██║  ██║██║  ██║   ██║
 ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝
"""


def show_banner() -> None:
    console.print(Text(_BANNER, style=f"bold {C_ACCENT}"))
    console.print(Text("  gUrrT · Video RAG CLI", style=f"#ff9900"))
    console.print(Rule(style=BORDER_PRIMARY))


def info(msg: str) -> None:
    console.print(f"[info]  {msg}[/info]")


def success(msg: str) -> None:
    console.print(f"[success]  {msg}[/success]")


def warn(msg: str) -> None:
    console.print(f"[warning]  {msg}[/warning]")


def error(msg: str) -> None:
    console.print(f"[error]  {msg}[/error]")


def step(msg: str) -> None:
    console.print(f"[primary]→[/primary]  {msg}")


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(style=BORDER_PRIMARY),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style=BORDER_PRIMARY, complete_style=C_ACCENT),
        TextColumn("[dim]{task.percentage:>3.0f}%[/dim]"),
        TimeElapsedColumn(),
        console=console,
    )
