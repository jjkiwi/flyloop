import numpy as np
import pandas as pd
import pytest

from flyloop.connectome.schema import Connectome, build_signed_matrix
from flyloop.connectome.signs import sign_of, unknown_transmitters
from flyloop.connectome.synthetic import synthetic_connectome


def test_glutamate_is_inhibitory_in_drosophila():
    # The single most common sign bug in community reimplementations.
    assert sign_of("glutamate") == -1
    assert sign_of("GABA") == -1
    assert sign_of("acetylcholine") == +1


def test_neuromodulators_carry_no_fast_sign():
    for nt in ("dopamine", "octopamine", "serotonin"):
        assert sign_of(nt) == 0


def test_unknown_transmitters_are_reported_not_guessed():
    s = pd.Series(["acetylcholine", "kryptonite", "kryptonite"])
    assert sign_of("kryptonite") == 0
    assert unknown_transmitters(s).to_dict() == {"kryptonite": 2}


def test_signed_matrix_uses_presynaptic_transmitter():
    neurons = pd.DataFrame(
        {"id": [1, 2], "type": ["exc", "inh"], "nt": ["acetylcholine", "gaba"]}
    )
    edges = pd.DataFrame({"pre": [1, 2], "post": [2, 1], "weight": [7, 5]})
    W = build_signed_matrix(neurons, edges)
    assert W[0, 1] == pytest.approx(+7)
    assert W[1, 0] == pytest.approx(-5)


def test_edges_to_unknown_neurons_are_dropped():
    neurons = pd.DataFrame({"id": [1], "type": ["a"], "nt": ["acetylcholine"]})
    edges = pd.DataFrame({"pre": [1, 99], "post": [1, 1], "weight": [3, 4]})
    W = build_signed_matrix(neurons, edges)
    assert W.nnz == 1


def test_population_lookup_and_require():
    c = synthetic_connectome()
    assert len(c.population("DNp09")) == 2
    assert len(c.population(r"^DNa\d+$", regex=True)) == 4
    with pytest.raises(KeyError):
        c.require("NoSuchNeuron")


def test_subset_preserves_internal_edges():
    c = synthetic_connectome()
    idx = np.concatenate([c.population("DNp01"), c.population("TTMn")])
    sub = c.subset(idx)
    assert sub.n == len(idx)
    assert sub.n_connections > 0


def test_roundtrip_save_load(tmp_path):
    c = synthetic_connectome(n_central=50)
    c.save(tmp_path / "cx")
    back = Connectome.load(tmp_path / "cx")
    assert back.n == c.n
    assert back.n_connections == c.n_connections
    assert back.name == c.name


def test_shape_mismatch_is_rejected():
    import scipy.sparse as sp

    neurons = pd.DataFrame({"id": [1, 2], "type": ["a", "b"], "nt": ["gaba", "gaba"]})
    with pytest.raises(ValueError):
        Connectome(neurons, sp.csr_matrix((3, 3)))


# ------------------------------------------- photoreceptors and their sign


def test_photoreceptors_labelled_histamine_pass():
    import pandas as pd

    from flyloop.connectome.signs import photoreceptor_check

    n = pd.DataFrame(
        {"type": ["R1-6", "R7", "R8", "L1"], "nt": ["histamine"] * 3 + ["acetylcholine"]}
    )
    out = photoreceptor_check(n)
    assert out["ok"] and out["n"] == 3 and out["wrong"] == 0


def test_a_photoreceptor_called_cholinergic_is_flagged():
    """FlyWire's transmitter prediction has no histamine class, so its R1-6 come
    back as acetylcholine. Left unflagged that inverts the visual system's first
    and largest synapse."""
    import pandas as pd

    from flyloop.connectome.signs import photoreceptor_check

    n = pd.DataFrame(
        {"type": ["R1-6", "R1-6", "R7"], "nt": ["acetylcholine", "acetylcholine", "glutamate"]}
    )
    out = photoreceptor_check(n)
    assert not out["ok"]
    assert out["wrong"] == 3
    assert "inverted" in out["note"]
    assert out["by_nt"] == {"acetylcholine": 2, "glutamate": 1}


def test_a_dataset_with_no_photoreceptors_is_not_flagged():
    """FlyWire's central-brain-only preparations have none, which is fine."""
    import pandas as pd

    from flyloop.connectome.signs import photoreceptor_check

    out = photoreceptor_check(pd.DataFrame({"type": ["DNa02"], "nt": ["acetylcholine"]}))
    assert out["ok"] and out["n"] == 0


def test_the_pattern_does_not_catch_unrelated_types():
    """Plenty of cell types start with R; only R1..R8 are photoreceptors."""
    import pandas as pd

    from flyloop.connectome.signs import photoreceptor_check

    n = pd.DataFrame(
        {"type": ["Rostrum", "R9x", "RIM", "R1-6"], "nt": ["acetylcholine"] * 3 + ["histamine"]}
    )
    assert photoreceptor_check(n)["n"] == 1


def test_cell_identity_overrides_a_wrong_photoreceptor_transmitter():
    """R1-6 releases histamine whatever a transmitter classifier says."""
    import pandas as pd

    from flyloop.connectome.signs import sign_vector

    nt = pd.Series(["acetylcholine", "glutamate", "acetylcholine"])
    types = pd.Series(["R1-6", "R7", "L1"])
    assert sign_vector(nt, types).tolist() == [-1, -1, +1]


def test_without_types_the_transmitter_is_taken_at_face_value():
    """The override is opt-in, so existing callers keep their old behaviour."""
    import pandas as pd

    from flyloop.connectome.signs import sign_vector

    nt = pd.Series(["acetylcholine", "glutamate"])
    assert sign_vector(nt).tolist() == [+1, -1]


def test_the_override_changes_nothing_when_the_label_was_already_right():
    import pandas as pd

    from flyloop.connectome.signs import sign_vector

    nt = pd.Series(["histamine", "histamine"])
    types = pd.Series(["R1-R6", "R7y"])
    assert sign_vector(nt, types).tolist() == sign_vector(nt).tolist() == [-1, -1]


def test_only_photoreceptors_are_overridden():
    """A general licence to override transmitters by name would be dangerous."""
    import pandas as pd

    from flyloop.connectome.signs import sign_vector

    nt = pd.Series(["acetylcholine"] * 4)
    types = pd.Series(["Rostrum", "RIM", "R9", "DNa02"])
    assert sign_vector(nt, types).tolist() == [+1, +1, +1, +1]
