from experiment.task import Experiment
from experiment.trial_generator import TrialSpec
from experiment.logging_io import GAZE_FIELDS
from hardware.mock_eye_tracker import MockTracker
from hardware.mock_markers import MockMarkers


class _Rows:
    def __init__(self):
        self.rows = []

    def log_gaze(self, row):
        self.rows.append(row)


def test_log_gaze_writes_trial_tagged_row():
    et = MockTracker()
    exp = Experiment({}, participant_id="1", session_id="1", eye_tracker=et, markers=MockMarkers())
    exp.logger = _Rows()
    trial = TrialSpec(2, "reversal", 5, 165, 5, False, 3, "vertical", True, 2, True, 2, "j")
    assert exp._log_gaze(trial) is None  # not recording → nothing logged
    et.initialize(); et.start_recording()
    assert exp._log_gaze(trial) is not None
    (row,) = exp.logger.rows
    assert set(row) == set(GAZE_FIELDS)
    assert (row["trial_id"], row["block"], row["phase"]) == (165, 2, "reversal")
