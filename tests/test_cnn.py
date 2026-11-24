import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import f1_score  # noqa: E402

from predmaint.cnn import BearingCNN, CNNClassifier  # noqa: E402
from predmaint.simulate import SimConfig, make_dataset  # noqa: E402


@pytest.fixture(scope="module")
def data():
    cfg = SimConfig(severity_range=(1.0, 2.0), noise_std_range=(0.2, 0.4))
    return make_dataset(40, cfg, seed=1), make_dataset(15, cfg, seed=2)


def test_forward_shape_and_parameter_count():
    net = BearingCNN(n_classes=4)
    out = net(torch.zeros(3, 1, 12000))
    assert out.shape == (3, 4)
    assert 5_000 < CNNClassifier().n_parameters < 100_000


def test_learns_better_than_chance(data):
    tr, te = data
    m = CNNClassifier(epochs=12, seed=0).fit(tr.signals, tr.labels)
    assert m.history[-1] < m.history[0]  # loss goes down
    assert f1_score(te.labels, m.predict(te.signals), average="macro") > 0.5  # chance is 0.25


def test_probabilities_and_determinism(data):
    tr, te = data
    a = CNNClassifier(epochs=2, seed=3).fit(tr.signals[:60], tr.labels[:60]).predict_proba(te.signals[:10])
    b = CNNClassifier(epochs=2, seed=3).fit(tr.signals[:60], tr.labels[:60]).predict_proba(te.signals[:10])
    assert a.shape == (10, 4) and np.allclose(a.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b, atol=1e-5)


def test_rejects_bad_labels_and_unfitted_use(data):
    tr, _ = data
    with pytest.raises(ValueError):
        CNNClassifier().fit(tr.signals[:8], np.full(8, 7))
    with pytest.raises(RuntimeError):
        CNNClassifier().predict(tr.signals[:2])
