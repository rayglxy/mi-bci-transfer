"""BCI Competition IV dataset 2a (BNCI 2014-001): loading and epoching.

Nine people, two recording days each ("T" and "E"), four imagined movements,
22 EEG channels at 250 Hz, 288 trials per day. Each .mat file holds nine runs;
the first three are eye-movement calibration and carry no trials.
"""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

FS = 250
CLASSES = ('left hand', 'right hand', 'feet', 'tongue')
SUBJECTS = tuple(range(1, 10))
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
URL = 'https://bnci-horizon-2020.eu/database/data-sets/001-2014/'


def load_session(subject, session, band=(8.0, 30.0), tmin=2.5, tmax=5.5, data_dir=DATA):
    """Epochs for one person and one day.

    Times are measured from trial start; the cue appears at 2 s and imagery runs
    to 6 s, so the default window is 0.5 to 3.5 s after the cue.
    Returns X (trials x 22 channels x samples), y (0..3) and the run of each trial.
    """
    m = sio.loadmat(os.path.join(data_dir, f'A0{subject}{session}.mat'), squeeze_me=True, struct_as_record=False)
    sos = butter(4, band, btype='band', fs=FS, output='sos')
    a, b = int(tmin * FS), int(tmax * FS)
    X, y, run = [], [], []
    for r, block in enumerate(np.atleast_1d(m['data'])):
        onsets = np.atleast_1d(block.trial)
        if onsets.size == 0:
            continue
        eeg = sosfiltfilt(sos, block.X[:, :22].T.astype(float), axis=1)     # filter each run on its own
        for o, label in zip(onsets.astype(int) - 1, np.atleast_1d(block.y)):   # MATLAB indices start at 1
            X.append(eeg[:, o + a:o + b]); y.append(int(label) - 1); run.append(r)
    return np.stack(X), np.array(y), np.array(run)


def load_all(subjects=SUBJECTS, **kw):
    """{(subject, session): (X, y, run)} for every recording."""
    return {(s, k): load_session(s, k, **kw) for s in subjects for k in 'TE'}
