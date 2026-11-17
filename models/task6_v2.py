"""Task 6 v2: predict the time until each user's final interaction.

The leaderboard metric behaves like MAE (the user-median baseline scores ~12M
seconds of MAE locally), so the model predicts conditional medians
(absolute-error gradient boosting) instead of means.

Features on top of task6_advanced.features():
* Reading dates of the known next book relative to the user's last activity
  (quantiles of all dates, and of dates after the last activity), excluding the
  query user's own interaction with that book.
* events_after: how many more interactions follow the predicted one (0 at test).
* to_end_days: time left until the end of the data.
Training rows are each user's last 30 gaps plus the users' final gaps (weight 5),
which match the test situation exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DAY = 86400.
FINAL_GAP_WEIGHT = 5.

def _load_advanced(data_dir):
    from . import task6_advanced
    return task6_advanced

class ItemDates:
    """Distribution of a book's reading dates relative to a reference time."""
    columns = ['item_n', 'item_frac_after', 'item_q02_rel', 'item_q50_rel', 'item_q98_rel',
               'after_q10', 'after_q25', 'after_q50', 'after_q75', 'after_q90', 'after_n']
    def fit(self, raw):
        ordered = raw[['user', 'item', 'timestamp']].sort_values('timestamp')
        self.times = {k: (v.timestamp.to_numpy(float), v.user.to_numpy()) for k, v in ordered.groupby('item', sort=False)}
        return self
    def transform(self, users, items, t_last):
        out = np.full((len(users), len(self.columns)), np.nan)
        for r, (u, i, t) in enumerate(zip(users, items, t_last)):
            entry = self.times.get(i)
            if entry is None: continue
            ts, us = entry; ts = ts[us != u]          # never use the user's own interaction
            if not len(ts): continue
            k = np.searchsorted(ts, t, 'right'); after = ts[k:]-t
            out[r, 0] = np.log1p(len(ts)); out[r, 1] = (len(ts)-k)/len(ts)
            out[r, 2:5] = (np.quantile(ts, [.02, .5, .98])-t)/DAY
            if len(after):
                out[r, 5:10] = np.quantile(after, [.1, .25, .5, .75, .9])/DAY
                out[r, 10] = np.log1p(len(after))
        return pd.DataFrame(out, columns=self.columns)

def make_model():
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=1500, learning_rate=.03, max_leaf_nodes=127,
                                         min_samples_leaf=300, l2_regularization=30, random_state=42)

def prepare(data_dir):
    adv = _load_advanced(data_dir)
    state = adv.prepare(data_dir)
    raw = state['raw']
    state['adv'] = adv
    state['item_dates'] = ItemDates().fit(raw)
    state['events_after'] = raw.groupby('user', sort=False).cumcount(ascending=False)-1
    state['t_end'] = float(np.quantile(raw.timestamp, .9999))
    return state

def _design(state, reference, rows, loo, items=None, final=False):
    raw = state['raw']
    base = state['adv'].features(state, reference, rows, items=items, loo=loo).reset_index(drop=True)
    next_items = raw.loc[rows, 'next_item'].to_numpy() if items is None else np.asarray(items)
    t_last = raw.loc[rows, 'timestamp'].to_numpy(float)
    extra = state['item_dates'].transform(raw.loc[rows, 'user'].to_numpy(), next_items, t_last)
    X = pd.concat([base, extra], axis=1)
    X['events_after'] = 0 if final else np.minimum(state['events_after'].loc[rows].to_numpy(), 30)
    X['to_end_days'] = (state['t_end']-t_last)/DAY
    return X

def _history_rows(state, reference):
    return state['raw'].loc[reference].groupby('user', sort=False).tail(30).index.to_numpy()

def evaluate(state):
    """Fit on history rows + tuning users' final gaps; report MAE on evaluation users."""
    raw = state['raw']; ref = state['train']; tune, ev = state['tune'], state['evaluation']
    hist = _history_rows(state, ref)
    X = pd.concat([_design(state, ref, hist, True), _design(state, ref, tune, False, final=True)], ignore_index=True)
    y = np.concatenate([raw.loc[hist, 'target'].to_numpy(float), raw.loc[tune, 'target'].to_numpy(float)])
    w = np.concatenate([np.ones(len(hist)), np.full(len(tune), FINAL_GAP_WEIGHT)])
    model = make_model().fit(X, y, sample_weight=w)
    ye = raw.loc[ev, 'target'].to_numpy(float)
    pred = np.maximum(model.predict(_design(state, ref, ev, False, final=True)), 0)
    fallback = raw.loc[ref, 'target'].median()
    baseline = state['x'].loc[ev, 'history_median'].fillna(fallback).to_numpy()
    results = pd.Series({'baseline user-median MAE': np.mean(np.abs(baseline-ye)),
                         'v2 model MAE': np.mean(np.abs(pred-ye))})
    print((results/1e6).round(3).to_string(), '(millions of seconds)', flush=True)
    return results

def predict(state, output_dir):
    raw = state['raw']
    all_ref = raw.index[state['valid']].to_numpy()
    hist = _history_rows(state, all_ref)
    final_rows = np.concatenate([state['tune'], state['evaluation']])
    X = pd.concat([_design(state, all_ref, hist, True), _design(state, state['train'], final_rows, False, final=True)], ignore_index=True)
    y = np.concatenate([raw.loc[hist, 'target'].to_numpy(float), raw.loc[final_rows, 'target'].to_numpy(float)])
    w = np.concatenate([np.ones(len(hist)), np.full(len(final_rows), FINAL_GAP_WEIGHT)])
    print('Fitting Task 6 v2 on', len(y), 'rows', flush=True)
    model = make_model().fit(X, y, sample_weight=w)
    test = pd.read_csv(state['data_dir']/'task_6_test.csv', dtype={'user': 'string', 'item': 'string'})
    lookup = raw.reset_index().groupby('user', sort=False).tail(1).set_index('user')['index']
    if test.user.map(lookup).isna().any(): raise ValueError('Unexpected unseen test user')
    rows = test.user.map(lookup).to_numpy(int)
    test['delta'] = np.maximum(model.predict(_design(state, all_ref, rows, False, items=test.item.to_numpy(), final=True)), 0)
    if not np.isfinite(test.delta).all(): raise ValueError('Non-finite predictions')
    Path(output_dir).mkdir(exist_ok=True)
    test[['user', 'item', 'delta']].to_csv(Path(output_dir)/'task_6_predictions.csv', index=False)
    print('Wrote', Path(output_dir)/'task_6_predictions.csv', flush=True)
    return test
