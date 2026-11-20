"""The CLI signature: an ASCII drawing in a box, once per command, on stderr.

The drawings and phrases are kept here exactly as defined: the text
of a triple-quoted literal below is the part of every box row to the right of
the text column, one line per row, trailing spaces removed (``box`` pads them
back). ``box`` puts the first phrase on the drawing's second row and
:data:`SEES_YOU` on its second-to-last row, in a text column
:data:`TEXT_COLUMN` characters wide, inside a border of ``#`` that is
:data:`WIDTH` characters wide.

This module imports nothing from the package: it is data and one renderer.
"""

from __future__ import annotations

import random

#: Every row of a box is this many characters wide, the border included.
WIDTH = 81

#: The width of the text column at the left of the drawing.
TEXT_COLUMN = 38

#: The fixed second phrase of every box.
SEES_YOU = "geoversegoddess sees you"

#: Each drawing's rows, right of the text column (40 characters at most).
DRAWINGS: dict[str, str] = {
    "coala": r"""
   .-"-.               .-"-.
  / .-. \   _.---._   / .-. \
 | (   ) |.'       '.| (   ) |
  \ '-' /  o       o  \ '-' /
   '-._|      ___      |_.-'
        \    (___)    /
         '.    V    .'
           '-.___.-'
""",
    "astronauta": r'''
       .-"""-.
      /  ___  \
     |  / _ \  |
     | | (_) | |
      \ '---' /
     .-'-----'-.
    /  [=] [=]  \
''',
    "ovni": r"""
          _.---._
        .'  o o  '.
   ___.'___________'.___
  '--._ o   o   o  _.--'
        '-._____.-'
         /  | |  \
        /   | |   \
""",
    "alien": r"""
     ##          ##
       ##      ##
     ##############
   ####  ######  ####
  ####################
  ##  ############  ##
  ##  ##        ##  ##
        ###  ###
""",
    "controle": r"""
    _______________________
   /   _            (Y)    \
  |  _| |_       (X)   (B)  |
  | |_   _|         (A)     |
  |   |_|     ___     ___   |
   \_________/   \___/   \_/
""",
    "terminal": r"""
   .-----------------------.
   | $ pyfs-matrix run     |
   | [##########] 100%     |
   | > all points done_    |
   '-----------------------'
        ___|_______|___
       '---------------'
""",
    "disquete": r"""
    ________________
   | |  ________  | |
   | | |        | | |
   | | |  pyfs  | | |
   | |  --------  | |
   |    ______      |
   |   | |    |     |
   |___|_|____|_____|
""",
    "oculos": r'''
        .-"""""""-.
      .'           '.
     /  ___     ___  \
    |==(___)===(___)==|
    |                 |
     \   '._____.'   /
      '.           .'
        '-._____.-'
''',
    "joinha": r"""
          _
         ( )
     .---' '-.
    (___      '----.
    (___           |
    (___           |
     (__.__________'
""",
    "gameover": r"""
   .-------------------------.
   |                         |
   |    G A M E   O V E R    |
   |                         |
   |   continue?  [Y]  [N]   |
   '-------------------------'
""",
    "satelite": r"""
   [###]-o-[###]      .
        / \   \          *
       /   \   \~~
      ' ...  '   ~~ . .
     (  bzzt  )     ~~~
""",
    "segfault": r"""
   .-----------------------.
   | $ pyfs-matrix run     |
   | Traceback (...)       |
   | Error: see the log    |
   '-----------------------'
        ___|_______|___
       '---------------'
""",
    "robotriste": r"""
        .-----.
       | x   x |
       |  ___  |
       | /   \ |
       '-------'
      /|  [ ]  |\
     ' |_______| '
        |_|   |_|
""",
    "triste": r'''
        .-"""""""-.
      .'           '.
     /    O     O    \
    |                 |
    |      _____      |
     \   .'     '.   /
      '.           .'
        '-._____.-'
''',
    "explodiu": r'''
        \  |  /  *
     *  .-""""-.  .
      .'  \  /  '.
     /    x  x    \
    |              |
    |    .----.    |
     \   '----'   /
      '-.______.-'
''',
    "pressstart": r"""
    _______________________
   /   _            (Y)    \
  |  _| |_  PRESS  (X)  (B) |
  | |_   _| START     (A)   |
  |   |_|     ___     ___   |
   \_________/   \___/   \_/
""",
    "esc": r"""
    .-----------.
    |           |
    |    ESC    |
    |           |
    '-----------'
     \_________/
""",
    "retorno": r"""
          _.---._
        .'  o o  '.
   ___.'___________'.___
  '--._   <<< RTB  _.--'
        '-._____.-'
          .  :  .
         .   :   .
""",
    "sono": r'''
                 Z
        .-"""""-.   z
      .'         '.  z
     /  ---   ---  \
    |               |
    |      ___      |
     \    (___)    /
      '-._______.-'
''',
    "maoparada": r"""
         _  _  _
        | || || | _
        | || || || |
     _  |          |
    \ \ |          |
     \ \|          |
      \            /
       '----------'
""",
}

#: The phrases each drawing may carry; one is drawn at random.
PHRASES: dict[str, tuple[str, ...]] = {
    "coala": ("Let now COALA do its magic",),
    "astronauta": (
        "Houston, the run is in",
        "One small run, one giant polar",
        "Spacewalk complete. Data aboard",
    ),
    "ovni": ("Contact made. Data received", "They came for the polars", "Beamed up: every point"),
    "alien": (
        "Level cleared. Next wave incoming",
        "High score: zero failures",
        "Wave defeated. Load next matrix",
    ),
    "controle": ("Game saved. Progress kept", "Checkpoint reached", "Combo complete. Nice run"),
    "terminal": (
        "Compiled, tested, shipped",
        "Exit code 0. Life is good",
        "It works on this machine too",
    ),
    "disquete": (
        "Saved. No floppy was harmed",
        "Written to disk, 1.44 MB of joy",
        "Ctrl+S, but automatic",
    ),
    "oculos": (
        "Nailed it. Too cool for errors",
        "Smooth run. Shades on",
        "Deal with it: all converged",
    ),
    "joinha": ("Thumbs up. All green", "Approved by the solver", "Good to go"),
    "gameover": (
        "Not this time. The log knows why",
        "Game over. Insert log to continue",
        "Try again after a look at the log",
    ),
    "satelite": (
        "Signal lost. Read the log to reconnect",
        "Houston, the link went quiet",
        "No telemetry. The log has it",
    ),
    "segfault": (
        "Houston, we have a problem",
        "Traceback delivered. Read it",
        "Something broke. The log says what",
    ),
    "robotriste": (
        "The robot tried. The log explains",
        "Beep... boop... error",
        "Even robots have bad days",
    ),
    "triste": (
        "Oops. That did not go as planned",
        "Not the result we wanted",
        "Deep breath. Read the log",
    ),
    "explodiu": (
        "Mind blown. The log has answers",
        "Well, that was unexpected",
        "Kaboom. Details in the log",
    ),
    "pressstart": (
        "Stopped. Press START when ready",
        "Paused at your command",
        "Game paused. Resume when ready",
    ),
    "esc": ("Escaped. The log shows where", "ESC pressed. Run stopped", "Out, as you asked"),
    "retorno": ("Mission aborted. Crew back safe", "Returning to base", "Abort accepted"),
    "sono": ("Zzz. Stopped for now", "Taking a nap. Wake me later", "Resting. Resume when ready"),
    "maoparada": ("Hold on. Cancelled", "Stop acknowledged", "Halted at your command"),
}

#: The drawings each outcome draws from. ``post`` is a successful post,
#: by the ``post`` subcommand or by the post a ``collect`` runs.
OUTCOME_DRAWINGS: dict[str, tuple[str, ...]] = {
    "post": ("coala",),
    "success": (
        "astronauta",
        "ovni",
        "alien",
        "controle",
        "terminal",
        "disquete",
        "oculos",
        "joinha",
    ),
    "failed": ("gameover", "satelite", "segfault", "robotriste", "triste", "explodiu"),
    "cancelled": ("pressstart", "esc", "retorno", "sono", "maoparada"),
}


def rows(name: str) -> list[str]:
    """Return the drawing's rows, right of the text column, exactly as defined."""
    return DRAWINGS[name].split("\n")[1:-1]


def box(name: str, phrase: str) -> str:
    """Render one drawing with its phrase, as the lines of a box, no trailing newline.

    Parameters
    ----------
    name : str
        A key of :data:`DRAWINGS`.
    phrase : str
        The first phrase, at most :data:`TEXT_COLUMN` characters; nothing is
        truncated and nothing widens, so a longer one is refused.
    """
    if len(phrase) > TEXT_COLUMN:
        raise ValueError(
            f"the phrase {phrase!r} has {len(phrase)} characters and the text column "
            f"holds {TEXT_COLUMN}; shorten the phrase, the box does not widen"
        )
    art = rows(name)
    inner = WIDTH - 2
    border = "#" * WIDTH
    blank = "#" + " " * inner + "#"
    lines = [border, blank]
    for index, row in enumerate(art):
        text = phrase if index == 1 else SEES_YOU if index == len(art) - 2 else ""
        lines.append("#" + (" " + text.ljust(TEXT_COLUMN) + row).ljust(inner) + "#")
    lines += [blank, border]
    return "\n".join(lines)


def pick(outcome: str, chooser: random.Random) -> tuple[str, str]:
    """Draw a drawing of ``outcome`` and one of its phrases, with ``chooser``."""
    name = chooser.choice(OUTCOME_DRAWINGS[outcome])
    return name, chooser.choice(PHRASES[name])
