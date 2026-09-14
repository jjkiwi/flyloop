"""End-to-end tests. Kept short on purpose; the long runs live in the CLI."""

import pytest

from flyloop.body import Arena, KinematicBody, Pillar, cluttered_arena, looming_arena
from flyloop.connectome.synthetic import synthetic_connectome
from flyloop.experiments import looming_experiment, run_baseline
from flyloop.loop import ClosedLoop
from flyloop.motor import LocomotorCommand


@pytest.fixture(scope="module")
def cx():
    return synthetic_connectome()


def test_approaching_object_grows_in_the_visual_field():
    body = KinematicBody(looming_arena())
    sizes = []
    for _ in range(4):
        sizes.append(float((body.observe() < 0.5).mean()))
        for _ in range(50):
            body.step(LocomotorCommand())
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_body_reset_restores_obstacles():
    body = KinematicBody(looming_arena())
    start = body.state()["nearest_obstacle"]
    for _ in range(100):
        body.step(LocomotorCommand())
    assert body.state()["nearest_obstacle"] < start
    body.reset()
    assert body.state()["nearest_obstacle"] == pytest.approx(start)


def test_forward_command_moves_the_body_forward():
    body = KinematicBody(Arena())
    for _ in range(100):
        body.step(LocomotorCommand(forward=1.0))
    assert body.state()["x"] > 0


def test_turn_command_changes_heading():
    body = KinematicBody(Arena())
    for _ in range(50):
        body.step(LocomotorCommand(turn=1.0))
    assert body.state()["theta"] != 0.0


def test_closed_loop_runs_and_logs(cx):
    body = KinematicBody(looming_arena(), dt=0.01)
    loop = ClosedLoop(cx, body, seed=3)
    res = loop.run(0.3)
    assert len(res.log) == 30
    for col in ("forward", "turn", "stop", "escape", "spikes", "t"):
        assert col in res.log.columns
    assert res.spike_counts.shape == (cx.n,)


def test_spike_counts_cover_every_brain_step(cx):
    """The per-neuron counts must match the per-step totals, not the last step."""
    body = KinematicBody(looming_arena(), dt=0.01)
    loop = ClosedLoop(cx, body, seed=3)
    res = loop.run(0.5)
    assert res.spike_counts.sum() == res.log["spikes"].sum()


def test_loop_reset_returns_to_a_silent_start(cx):
    body = KinematicBody(looming_arena(), dt=0.01)
    loop = ClosedLoop(cx, body, seed=3)
    loop.run(0.5)
    loop.reset()
    assert loop.command.escape is False
    assert loop.body.state()["t"] == 0.0


def test_looming_triggers_escape_and_a_static_object_does_not(cx):
    res = looming_experiment(cx, n_trials=1, conditions=("looming", "static"))
    assert res.escape_rate["looming"] == 1.0
    assert res.escape_rate["static"] == 0.0
    assert res.passed


def test_experiment_flags_a_silent_network(cx):
    from flyloop.brain.lif import LIFParams

    res = looming_experiment(
        cx, n_trials=1, conditions=("looming",), params=LIFParams(w_syn=1e-9)
    )
    assert not res.passed
    assert any("never spiked" in n for n in res.notes)


def test_baseline_also_discriminates_but_reacts_much_earlier():
    loom = run_baseline(looming_arena())
    static = run_baseline(Arena([Pillar(x=1.5, y=0.0, radius=0.3, height=0.8)]))
    assert loom["escape"].max() == 1.0
    assert static["escape"].max() == 0.0
    first = loom.index[loom["escape"] > 0][0]
    assert float(loom["t"].iloc[first]) < 1.0


def test_cluttered_arena_is_reproducible():
    a, b = cluttered_arena(seed=7), cluttered_arena(seed=7)
    assert [p.x for p in a.pillars] == [p.x for p in b.pillars]
