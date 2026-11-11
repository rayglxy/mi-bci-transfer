#!/usr/bin/env python3
"""Run every experiment and write data/results.json.

    python3 scripts/run.py

Needs the 18 .mat files in data/ (see README). About two minutes.
"""
import json, os, sys, warnings
import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mibci.data import load_session, SUBJECTS
from mibci.align import covariances
from mibci.models import MODELS
from mibci import experiments as E

COUNTS = [0, 4, 8, 16, 32, 64, 144]


def main():
    data = {}
    for s in SUBJECTS:
        for k in 'TE':
            X, y, run = load_session(s, k)
            data[s, k] = (covariances(X), y, run)
        print('loaded subject', s, flush=True)

    res = {'subjects': list(SUBJECTS), 'trials_per_session': int(len(data[1, 'T'][1])), 'models': {}}
    for name, make in MODELS.items():
        r = {k: [] for k in ('within', 'session_raw', 'session_aligned', 'subject_raw', 'subject_aligned')}
        for s in SUBJECTS:
            CT, yT, runT = data[s, 'T']; CE, yE, _ = data[s, 'E']
            others = [data[o, 'T'][:2] for o in SUBJECTS if o != s]
            r['within'].append(E.within_session(make, CT, yT, runT))
            r['session_raw'].append(E.cross_session(make, (CT, yT), (CE, yE), align=False))
            r['session_aligned'].append(E.cross_session(make, (CT, yT), (CE, yE), align=True))
            r['subject_raw'].append(E.cross_subject(make, others, (CE, yE), align=False))
            r['subject_aligned'].append(E.cross_subject(make, others, (CE, yE), align=True))
        r['p_session'] = float(wilcoxon(r['session_aligned'], r['session_raw']).pvalue)
        r['p_subject'] = float(wilcoxon(r['subject_aligned'], r['subject_raw']).pvalue)
        res['models'][name] = r
        print(name, {k: round(float(np.mean(v)), 3) for k, v in r.items()}, flush=True)

    make = MODELS['Tangent space + logistic regression']
    curve = [E.calibration_curve(make, data[s, 'T'][:2], data[s, 'E'][:2], COUNTS) for s in SUBJECTS]
    res['calibration'] = {str(n): [c[n] for c in curve] for n in COUNTS}
    curve = [E.calibration_curve_subject(make, [data[o, 'T'][:2] for o in SUBJECTS if o != s], data[s, 'E'][:2], COUNTS)
             for s in SUBJECTS]
    res['calibration_new_person'] = {str(n): [c[n] for c in curve] for n in COUNTS}
    n_test = res['trials_per_session']
    res['chance'] = 0.25
    res['chance_95'] = float(0.25 + 1.645 * np.sqrt(0.25 * 0.75 / n_test))     # one-sided binomial bound
    json.dump(res, open(os.path.join(ROOT, 'data', 'results.json'), 'w'), indent=1)
    for key in ('calibration', 'calibration_new_person'):
        print(key, {n: round(float(np.mean(v)), 3) for n, v in res[key].items()})


if __name__ == '__main__':
    main()
