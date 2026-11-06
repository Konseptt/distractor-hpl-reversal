import pandas as pd

from analysis.behavioral_qc import location_category


def test_location_category_tracks_both_hpls():
    df = pd.DataFrame({
        "phase": ["acquisition"] * 3 + ["reversal"] * 4,
        "current_HPL": [6, 6, 6, 2, 2, 2, 2],
        "distractor_present": [True, True, False, True, True, True, False],
        "distractor_location": [6, 2, "", 6, 2, 4, ""],
    })
    assert list(location_category(df)) == [
        "acq_HPL", "rev_HPL", "absent", "acq_HPL", "rev_HPL", "other_LP", "absent",
    ]


def test_location_category_stable_mode():
    df = pd.DataFrame({
        "phase": ["stable"] * 2, "current_HPL": [2, 2],
        "distractor_present": [True, True], "distractor_location": [2, 5],
    })
    assert list(location_category(df)) == ["acq_HPL", "other_LP"]
