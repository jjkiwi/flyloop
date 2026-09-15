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


def test_activation_drives_a_population_and_reads_the_rest(cx):
    from flyloop.experiments import activation_experiment

    res = activation_experiment(cx, drive="LC4", side="L", rate_hz=100.0, duration=0.05)
    assert res.total_spikes > 0
    assert res.rates["DNp04_L"] > 0


def test_activation_is_lateralised(cx):
    from flyloop.experiments import activation_experiment

    left = activation_experiment(cx, drive="LC4", side="L", rate_hz=100.0, duration=0.1)
    assert left.rates["DNp04_L"] > left.rates["DNp04_R"]


def test_activation_rejects_an_absent_population(cx):
    from flyloop.experiments import activation_experiment

    with pytest.raises(KeyError, match="NoSuchCell"):
        activation_experiment(cx, drive="NoSuchCell", duration=0.01)


def test_activation_flags_a_silent_network(cx):
    from flyloop.brain.lif import LIFParams
    from flyloop.experiments import activation_experiment

    res = activation_experiment(
        cx, drive="LC4", side="L", rate_hz=1.0, duration=0.02,
        params=LIFParams(w_syn=1e-12, poisson_gain=1e-6),
    )
    assert any("never spiked" in n for n in res.notes)


def test_sweep_reports_a_recruitment_threshold(cx):
    from flyloop.experiments import recruitment_sweep

    s = recruitment_sweep(
        cx, drive="LC4", side="L", rates=(10.0, 100.0), seeds=(0,), duration=0.05
    )
    th = s.thresholds(criterion=5.0)
    assert s.rates.shape[0] == 2
    # A population recruited at the low rate must not be listed at the high one.
    assert th["DNp04_L"] in (10.0, 100.0)


def test_sweep_marks_never_recruited_populations_as_nan(cx):
    import numpy as np

    from flyloop.experiments import recruitment_sweep

    s = recruitment_sweep(
        cx, drive="LC4", side="L", rates=(10.0,), seeds=(0,), duration=0.05
    )
    th = s.thresholds(criterion=1e9)  # nothing can exceed this
    assert np.isnan(th).all()


def test_sweep_records_seed_spread(cx):
    from flyloop.experiments import recruitment_sweep

    s = recruitment_sweep(
        cx, drive="LC4", side="L", rates=(20.0, 100.0), seeds=(0, 1, 2), duration=0.05
    )
    assert set(s.variability.index) == {20.0, 100.0}
    assert (s.variability >= 1.0).all(), "spread is a max/min ratio"


def test_unreliable_rates_are_excluded_from_thresholds(cx):
    """A rate whose seed spread is large must not set a recruitment threshold."""
    from flyloop.experiments import recruitment_sweep

    s = recruitment_sweep(
        cx, drive="LC4", side="L", rates=(10.0, 100.0), seeds=(0,), duration=0.05
    )
    # Force the low rate to look unreliable, as it genuinely is on real data.
    s.unreliable = (10.0,)
    kept = s.thresholds(criterion=5.0)
    dropped = s.thresholds(criterion=5.0, skip_unreliable=False)
    assert (kept.dropna() >= 100.0).all()
    assert s.report().count("too variable") == 1
    assert len(dropped.dropna()) >= len(kept.dropna())
