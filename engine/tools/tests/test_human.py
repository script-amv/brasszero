# SPDX-License-Identifier: Apache-2.0
import brass
import pytest

from brass_ai.__main__ import main
from brass_ai.human import HumanAgent, QuitGame


def scripted(lines):
    it = iter(lines)

    def inp(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    out = []
    return HumanAgent(input_fn=inp, output_fn=out.append), out


def _obs(seed=5):
    g = brass.Game(seed)
    return g.observation(g.current_player())


def _group_of(obs, word):
    groups = []
    for i in range(obs.num_legal()):
        w = obs.describe_move(i).split()[0]
        if w not in groups:
            groups.append(w)
    return groups.index(word) + 1


def test_human_picks_group_then_move():
    obs = _obs()
    passes = [i for i in range(obs.num_legal()) if obs.describe_move(i).startswith("pass")]
    agent, out = scripted([str(_group_of(obs, "pass")), "1"])
    assert agent.act(obs) == passes[0]
    assert any("pass" in line for line in out)


def test_human_reprompts_on_bad_input():
    obs = _obs()
    prompts = []
    lines = iter(["", "abc", "999", str(_group_of(obs, "pass")), "1"])

    def inp(prompt=""):
        prompts.append(prompt)
        return next(lines)

    agent = HumanAgent(input_fn=inp, output_fn=lambda *_: None)
    i = agent.act(obs)
    assert obs.describe_move(i).startswith("pass")
    assert len(prompts) == 5


def test_human_filter_and_back():
    obs = _obs()
    n_build = sum(1 for i in range(obs.num_legal()) if obs.describe_move(i).startswith("build"))
    builds = [obs.describe_move(i) for i in range(obs.num_legal()) if obs.describe_move(i).startswith("build")]
    loc = builds[0].split(" at ")[1].split(" ")[0]
    agent, out = scripted([str(_group_of(obs, "build")), "/" + loc, "b", str(_group_of(obs, "pass")), "1"])
    i = agent.act(obs)
    assert obs.describe_move(i).startswith("pass")
    cut = out.index("Moves matching %r:" % loc)

    def builds(lines):
        return [l for l in lines if l.startswith("  ") and ") build " in l and "(discard" in l]

    assert len(builds(out[:cut])) == n_build  # full list first
    narrowed = builds(out[cut:])
    assert 0 < len(narrowed) < n_build and all(" at %s " % loc in l for l in narrowed)
    assert "Choose a kind of move:" in out[cut:]  # "b" returned to the groups


def test_human_eof_quits():
    agent, _ = scripted([])
    with pytest.raises(QuitGame):
        agent.act(_obs())


def test_human_quit_and_reprint_board():
    agent, out = scripted(["?", "q"])
    with pytest.raises(QuitGame):
        agent.act(_obs())
    assert sum(1 for line in out if "era, round" in line) == 2


def test_play_command_quits_cleanly(capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    main(["play", "--seat", "0", "--opponents", "random", "random", "random", "--seed", "4"])
    out = capsys.readouterr().out
    assert "Game abandoned." in out
    assert "seed" not in out


def test_human_non_ascii_digit_reprompts():
    agent, out = scripted(["²", "٣", "q"])  # superscript two, Arabic-Indic three
    with pytest.raises(QuitGame):
        agent.act(_obs())
    assert sum(1 for line in out if line.startswith("Enter a number")) == 2


def test_play_command_ctrl_c_abandons(capsys, monkeypatch):
    class Interrupting:
        def readline(self, *a):
            raise KeyboardInterrupt

        def read(self, *a):
            raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin", Interrupting())
    main(["play", "--seat", "0", "--opponents", "random", "random", "random", "--seed", "4"])
    out = capsys.readouterr().out
    assert "Game abandoned." in out and "seed" not in out
