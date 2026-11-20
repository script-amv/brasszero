# SPDX-License-Identifier: Apache-2.0
"""A human seat for terminal play. Sees only the Observation it is handed."""
from typing import Callable, List, Optional, Tuple


class QuitGame(Exception):
    """The human quit (or stdin ended)."""


class HumanAgent:
    name = "human"

    def __init__(self, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print):
        self._in = input_fn
        self._out = output_fn

    def _ask(self, prompt: str) -> str:
        try:
            return self._in(prompt).strip()
        except EOFError:
            raise QuitGame() from None

    def act(self, obs) -> int:
        out = self._out
        out(obs.render())
        descs = [obs.describe_move(i) for i in range(obs.num_legal())]
        groups: List[str] = []
        for d in descs:
            w = d.split()[0]
            if w not in groups:
                groups.append(w)

        group: Optional[str] = groups[0] if len(groups) == 1 else None
        text = ""
        shown: List[Tuple[int, str]] = []
        dirty = True
        while True:
            if group is None:
                if dirty:
                    out("Choose a kind of move:")
                    for n, g in enumerate(groups, 1):
                        out("  %d) %s (%d)" % (n, g, sum(1 for d in descs if d.split()[0] == g)))
                    dirty = False
                reply = self._ask("group> ")
            else:
                if dirty:
                    shown = [(i, d) for i, d in enumerate(descs) if d.split()[0] == group and text in d]
                    if text:
                        out("Moves matching %r:" % text)
                    for n, (_, d) in enumerate(shown, 1):
                        out("  %d) %s" % (n, d))
                    if not shown:
                        out("  (no moves match; try another filter or b)")
                    dirty = False
                reply = self._ask("move> ")

            if reply == "q":
                raise QuitGame()
            if reply == "?":
                out(obs.render())
                dirty = True
            elif reply == "b":
                if len(groups) > 1:
                    group, text = None, ""
                dirty = True
            elif reply.isascii() and reply.isdecimal():
                n = int(reply)
                if group is None:
                    if 1 <= n <= len(groups):
                        group, text, dirty = groups[n - 1], "", True
                        continue
                elif 1 <= n <= len(shown):
                    return shown[n - 1][0]
                out("Not a valid choice.")
            elif reply.startswith("/") and group is not None:
                text, dirty = reply[1:], True
            else:
                out("Enter a number, /text to filter, b to go back, ? for the board, q to quit.")
