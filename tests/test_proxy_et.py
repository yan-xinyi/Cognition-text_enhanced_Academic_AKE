import numpy as np
import pandas as pd

from cognition_ake.proxy_et import ETScaler, document_windows, eligible_word, select_owner_windows


def test_windows_cover_once_by_owner():
    owner = select_owner_windows(700, width=256, stride=192)
    assert owner.shape == (700,)
    assert np.all(owner >= 0)
    assert document_windows(0) == []


def test_scaler_uses_five_targets():
    frame = pd.DataFrame({
        "nFix": [1, 2], "FFD": [2, 4], "GPT": [3, 6], "TRT": [4, 8], "fixProp": [5, 10]
    })
    scaler = ETScaler.fit(frame)
    transformed = scaler.transform(frame.to_numpy(float))
    assert transformed.shape == (2, 5)
    assert np.allclose(transformed.mean(axis=0), 0.0)


def test_eligible_word_policy():
    assert eligible_word("control")
    assert not eligible_word("---")

