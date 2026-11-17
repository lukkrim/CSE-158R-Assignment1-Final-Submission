"""Task 6 v3: task6_v2 plus co-reader and joint-rhythm features, 3-model average.

New features for a query (user u, last activity t_last, next book B):
* CoReaders: for other users who read both u's latest book(s) A and B, the time
  between their reads of A and B.
* CoReadEstimate: per co-reader, the implied gap t_u(A) + (t_v(B) - t_v(A)) - t_last,
  over u's last 10 books.
* Near-end history gaps (1-3 reads before the user's last) are up-weighted.
* JointGap: B's reading dates after t_last, weighted by how plausible each gap is
  under u's own (log-scale) gap histogram.
The final prediction averages three absolute-error boosters (conditional medians).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DAY = 86400.
FINAL_GAP_WEIGHT = 5.

def _load(data_dir, name):
    import importlib
    return importlib.import_module(f'.{name}', __package__)

class CoReaders:
    """For readers of both a recent book A and the next book B (excluding the user):
    the time between their reads of A and B."""
    cols=['co_n1','co_med1','co_pos1','co_posmed1','co_n3','co_med3','co_posmed3']
    def fit(self, raw):
        self.read={i:dict(zip(g.user,g.timestamp)) for i,g in raw.groupby('item',sort=False)}
        self.raw=raw; self.start=np.arange(len(raw))-raw.groupby('user',sort=False).cumcount().to_numpy()
        return self
    def _deltas(self,a,b,u):
        ra,rb=self.read.get(a),self.read.get(b)
        if ra is None or rb is None: return np.array([])
        if len(ra)>len(rb): common=[v for v in rb if v in ra and v!=u]
        else: common=[v for v in ra if v in rb and v!=u]
        return np.array([rb[v]-ra[v] for v in common],float)
    def transform(self, rows, next_items):
        items=self.raw.item.to_numpy(); users=self.raw.user.to_numpy(); out=np.full((len(rows),len(self.cols)),np.nan)
        for k,(r,b) in enumerate(zip(rows,next_items)):
            u=users[r]; d=self._deltas(items[r],b,u)
            out[k,0]=np.log1p(len(d))
            if len(d):
                out[k,1]=np.median(d)/DAY; pos=d[d>0]; out[k,2]=len(pos)/len(d)
                if len(pos): out[k,3]=np.median(pos)/DAY
            ds=[d]+[self._deltas(items[j],b,u) for j in range(max(self.start[r],r-2),r)]
            d3=np.concatenate(ds); out[k,4]=np.log1p(len(d3))
            if len(d3):
                out[k,5]=np.median(d3)/DAY; pos=d3[d3>0]
                if len(pos): out[k,6]=np.median(pos)/DAY
        return pd.DataFrame(out,columns=self.cols)

class CoReadEstimate(CoReaders):
    """Direct gap estimates: t_u(A) + (t_v(B) - t_v(A)) - t_last over co-readers v of
    each of the user's last 10 books A and the next book B."""
    cols=['est_n','est_q25','est_q50','est_q75','est_pos_q50','est_recent_q50']
    def transform(self, rows, next_items):
        items=self.raw.item.to_numpy(); users=self.raw.user.to_numpy(); ts=self.raw.timestamp.to_numpy(float)
        out=np.full((len(rows),len(self.cols)),np.nan)
        for k,(r,b) in enumerate(zip(rows,next_items)):
            u=users[r]; tl=ts[r]; ests=[]; recent=[]
            for j in range(max(self.start[r],r-9),r+1):
                d=self._deltas(items[j],b,u)
                if len(d):
                    e=ts[j]+d-tl; ests.append(e)
                    if j>=r-2: recent.append(e)
            out[k,0]=np.log1p(sum(len(e) for e in ests))
            if ests:
                e=np.concatenate(ests); out[k,1:4]=np.quantile(e,[.25,.5,.75])/DAY
                pos=e[e>0]
                if len(pos): out[k,4]=np.median(pos)/DAY
                if recent: out[k,5]=np.median(np.concatenate(recent))/DAY
        return pd.DataFrame(out,columns=self.cols)

NB=48; LO,HI=0.,np.log1p(4e9)
def bin_of(x): return np.clip(((np.log1p(np.maximum(x,0))-LO)/(HI-LO)*NB).astype(int),0,NB-1)
def weighted_quantiles(v,w,qs):
    o=np.argsort(v); v,w=v[o],w[o]; c=np.cumsum(w); c/=c[-1]
    return np.interp(qs,c,v)
class JointGap:
    """Combine the user's own gap distribution with the next book's reading dates."""
    def fit(self, raw, item_times):
        self.raw=raw; self.item_times=item_times
        g=raw.gap.to_numpy(float); valid=np.isfinite(g)&(g>=0)
        prior=np.bincount(bin_of(g[valid]),minlength=NB).astype(float); self.prior=prior/prior.sum()
        # cumulative per-row histogram of the user's gaps up to and including that row
        self.user_start=raw.groupby('user',sort=False).cumcount().to_numpy()
        return self
    def user_hist(self, rows):
        raw=self.raw; out=np.zeros((len(rows),NB))
        users=raw.user.to_numpy(); gaps=raw.gap.to_numpy(float)
        starts=np.arange(len(raw))-self.user_start
        for k,r in enumerate(rows):
            g=gaps[starts[r]:r+1]; g=g[np.isfinite(g)&(g>=0)]
            if len(g): out[k]=np.bincount(bin_of(g),minlength=NB)
        return out
    def transform(self, rows, items):
        raw=self.raw; H=self.user_hist(rows)
        kernel=np.array([.1,.2,.4,.2,.1])
        res=np.full((len(rows),6),np.nan)
        users=raw.user.to_numpy()[rows]; t=raw.timestamp.to_numpy(float)[rows]
        for k,(u,i,tt) in enumerate(zip(users,items,t)):
            h=np.convolve(H[k],kernel,'same'); n=H[k].sum()
            dens=(h+3*self.prior)/(n+3)                       # user's smoothed log-gap density
            entry=self.item_times.get(i)
            res[k,5]=np.log1p(n)
            if entry is None: continue
            ts,us=entry; ts=ts[us!=u]; a=ts[ts>tt]-tt
            if not len(a): continue
            w=dens[bin_of(a)]
            res[k,0:3]=weighted_quantiles(a,w,[.25,.5,.75])/DAY
            res[k,3]=np.log(w.sum()+1e-12)                    # how compatible the book is with the user's rhythm
            res[k,4]=np.average(np.log1p(a),weights=w)
        return pd.DataFrame(res,columns=['joint_q25','joint_q50','joint_q75','joint_logmass','joint_meanlog','user_ngaps'])

MODEL_CONFIGS = [dict(random_state=42),
                 dict(random_state=7, max_leaf_nodes=63, min_samples_leaf=200),
                 dict(random_state=123, max_leaf_nodes=255, min_samples_leaf=400, learning_rate=.02, max_iter=2500)]

def make_model(**overrides):
    params = dict(loss='absolute_error', max_iter=1500, learning_rate=.03, max_leaf_nodes=127,
                  min_samples_leaf=300, l2_regularization=30)
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)

def prepare(data_dir):
    v2 = _load(data_dir, 'task6_v2')
    state = v2.prepare(data_dir)
    raw = state['raw']
    state['v2'] = v2
    state['coreaders'] = CoReaders().fit(raw)
    state['coestimate'] = CoReadEstimate().fit(raw)
    state['joint'] = JointGap().fit(raw, state['item_dates'].times)
    return state

def _design(state, reference, rows, loo, items=None, final=False):
    raw = state['raw']
    X = state['v2']._design(state, reference, rows, loo, items=items, final=final)
    next_items = raw.loc[rows, 'next_item'].to_numpy() if items is None else np.asarray(items)
    parts = [X, state['coreaders'].transform(rows, next_items), state['joint'].transform(rows, next_items),
             state['coestimate'].transform(rows, next_items)]
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

def _history_weights(X_hist):
    # gaps followed by only 1-3 more reads are closest to the (final) test situation
    after = X_hist.events_after.to_numpy()
    return np.where(after <= 1, 3., np.where(after <= 3, 2., 1.))

def _fit_ensemble(X, y, w):
    return [make_model(**cfg).fit(X, y, sample_weight=w) for cfg in MODEL_CONFIGS]

def _predict(models, X):
    return np.maximum(np.mean([m.predict(X) for m in models], axis=0), 0)

def evaluate(state):
    raw = state['raw']; ref = state['train']; tune, ev = state['tune'], state['evaluation']
    hist = state['v2']._history_rows(state, ref)
    X = pd.concat([_design(state, ref, hist, True), _design(state, ref, tune, False, final=True)], ignore_index=True)
    y = np.concatenate([raw.loc[hist, 'target'].to_numpy(float), raw.loc[tune, 'target'].to_numpy(float)])
    w = np.concatenate([_history_weights(X.iloc[:len(hist)]), np.full(len(tune), FINAL_GAP_WEIGHT)])
    models = _fit_ensemble(X, y, w)
    ye = raw.loc[ev, 'target'].to_numpy(float)
    pred = _predict(models, _design(state, ref, ev, False, final=True))
    baseline = state['x'].loc[ev, 'history_median'].fillna(raw.loc[ref, 'target'].median()).to_numpy()
    results = pd.Series({'baseline user-median MAE': np.mean(np.abs(baseline-ye)), 'v3 ensemble MAE': np.mean(np.abs(pred-ye))})
    print((results/1e6).round(3).to_string(), '(millions of seconds)', flush=True)
    return results

def predict(state, output_dir):
    raw = state['raw']; v2 = state['v2']
    all_ref = raw.index[state['valid']].to_numpy()
    hist = v2._history_rows(state, all_ref)
    final_rows = np.concatenate([state['tune'], state['evaluation']])
    X = pd.concat([_design(state, all_ref, hist, True), _design(state, state['train'], final_rows, False, final=True)], ignore_index=True)
    y = np.concatenate([raw.loc[hist, 'target'].to_numpy(float), raw.loc[final_rows, 'target'].to_numpy(float)])
    w = np.concatenate([_history_weights(X.iloc[:len(hist)]), np.full(len(final_rows), FINAL_GAP_WEIGHT)])
    print('Fitting Task 6 v3 ensemble on', len(y), 'rows', flush=True)
    models = _fit_ensemble(X, y, w)
    test = pd.read_csv(state['data_dir']/'task_6_test.csv', dtype={'user': 'string', 'item': 'string'})
    lookup = raw.reset_index().groupby('user', sort=False).tail(1).set_index('user')['index']
    if test.user.map(lookup).isna().any(): raise ValueError('Unexpected unseen test user')
    rows = test.user.map(lookup).to_numpy(int)
    test['delta'] = _predict(models, _design(state, all_ref, rows, False, items=test.item.to_numpy(), final=True))
    if not np.isfinite(test.delta).all(): raise ValueError('Non-finite predictions')
    Path(output_dir).mkdir(exist_ok=True)
    test[['user', 'item', 'delta']].to_csv(Path(output_dir)/'task_6_predictions.csv', index=False)
    print('Wrote', Path(output_dir)/'task_6_predictions.csv', flush=True)
    return test
