"""Shared rating-prediction stack for Tasks 1 and 3.

Key observation: the hidden test sets contain no 0-star ratings (a 0 on Goodreads
means "shelved, not rated"). Local MSE on non-zero holdout rows matches the
leaderboard closely, so every model here is trained and validated on rating > 0.

Base models (kNN on bias residuals, implicit ALS, full-implicit SVD++, Bayesian FM,
timeSVD++, ...) become columns of a feature table; a seed-averaged gradient-boosted
regressor stacks them (validate_rating_task / predict_rating_task).
"""
import contextlib
import io

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit import als as implicit_als
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from .rating_base import ResidualNeighborhood1, fit_history, fit_svd, history_features, predict_svd
from .temporal_rating import EnhancedTimeFeatures3, TargetedFeatures3, Task3Features


def topk_graph(B, k, shrink, metric='jaccard', chunk=256):
    """B: entities x context binary matrix. Returns a top-k shrunk similarity graph."""
    B = B.tocsr().astype(np.float32); n = B.shape[0]
    count = np.asarray(B.sum(1)).ravel(); BT = B.T.tocsr()
    R, C, V = [], [], []
    for start in range(0, n, chunk):
        overlap = (B[start:start+chunk] @ BT).tocsr()
        for offset in range(overlap.shape[0]):
            i = start + offset; lo, hi = overlap.indptr[offset], overlap.indptr[offset+1]
            js, common = overlap.indices[lo:hi], overlap.data[lo:hi]
            keep = js != i; js, common = js[keep], common[keep]
            if not len(js): continue
            denominator = count[i]+count[js]-common if metric == 'jaccard' else np.sqrt(count[i]*count[js])
            w = common/np.maximum(denominator, 1)*common/(common+shrink)
            if len(w) > k:
                top = np.argpartition(w, -k)[-k:]; js, w = js[top], w[top]
            R.append(np.full(len(js), i)); C.append(js); V.append(w)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))

class KNNFeatures:
    """Item- and user-based kNN on bias residuals. Similarities come from ALL
    interactions (including 0-star shelvings and the unlabeled query pairs);
    residuals come from non-zero ratings only."""
    def fit(self, rated, interactions, k_item=(100, 400, 1500), k_user=(100, 400), cos_item=(400,)):
        U = pd.Index(pd.concat([rated.user, interactions.user]).unique())
        I = pd.Index(pd.concat([rated.item, interactions.item]).unique())
        self.U, self.I = U, I
        u, i, r = U.get_indexer(rated.user), I.get_indexer(rated.item), rated.rating.to_numpy(float)
        self.mu = r.mean(); nu, ni = len(U), len(I)
        bu, bi = np.zeros(nu), np.zeros(ni)
        for _ in range(15):
            bu = np.bincount(u, r-self.mu-bi[i], nu)/(np.bincount(u, minlength=nu)+5)
            bi = np.bincount(i, r-self.mu-bu[u], ni)/(np.bincount(i, minlength=ni)+10)
        self.bu, self.bi = bu, bi
        residual = r-self.mu-bu[u]-bi[i]
        self.Rres = sp.csr_matrix((residual, (u, i)), shape=(nu, ni))
        self.Rmask = sp.csr_matrix((np.ones_like(residual), (u, i)), shape=(nu, ni))
        self.Rraw = sp.csr_matrix((r, (u, i)), shape=(nu, ni))
        iu, ii = U.get_indexer(interactions.user), I.get_indexer(interactions.item)
        B = sp.csr_matrix((np.ones(len(iu), np.float32), (ii, iu)), shape=(ni, nu)); B.data[:] = 1
        self.graphs = {}
        for k in k_item: self.graphs['item', str(k)] = topk_graph(B, k, 20.)
        for k in cos_item: self.graphs['item', f'cos{k}'] = topk_graph(B, k, 20., 'cosine')
        for k in k_user: self.graphs['user', str(k)] = topk_graph(B.T.tocsr(), k, 20.)
        return self

    def transform(self, q):
        u, i = self.U.get_indexer(q.user), self.I.get_indexer(q.item)
        base = self.mu+np.where(u >= 0, self.bu[np.maximum(u, 0)], 0)+np.where(i >= 0, self.bi[np.maximum(i, 0)], 0)
        features = {'knn_base': base}
        rows = np.where((u >= 0) & (i >= 0))[0]
        for (kind, k), G in self.graphs.items():
            if kind == 'item':
                W = G[i[rows]].multiply(self.Rmask[u[rows]]).tocsr()
                res, raw = self.Rres[u[rows]], self.Rraw[u[rows]]
            else:
                W = G[u[rows]].multiply(self.Rmask.T.tocsr()[i[rows]]).tocsr()
                res, raw = self.Rres.T.tocsr()[i[rows]], self.Rraw.T.tocsr()[i[rows]]
            numerator = np.asarray(W.multiply(res).sum(1)).ravel()
            raw_numerator = np.asarray(W.multiply(raw).sum(1)).ravel()
            wsum = np.asarray(W.sum(1)).ravel(); n = np.diff(W.indptr)
            wmax = np.maximum.reduceat(np.append(W.data, 0), np.minimum(W.indptr[:-1], len(W.data))) if len(W.data) else np.zeros(len(rows))
            columns = {name: np.zeros(len(q)) for name in ('pred', 'wsum', 'n', 'raw', 'wmax')}
            columns['pred'][rows] = numerator/(wsum+.1); columns['wsum'][rows] = wsum
            columns['n'][rows] = n; columns['wmax'][rows] = np.where(n > 0, wmax, 0)
            columns['raw'][rows] = np.where(wsum > 0, raw_numerator/np.maximum(wsum, 1e-9), 0)
            tag = f'{kind}{k}'
            features[tag+'_pred'] = base+columns['pred']
            features[tag+'_wsum'] = np.log1p(columns['wsum']*100)
            features[tag+'_n'] = np.log1p(columns['n'])
            features[tag+'_raw'] = np.where(columns['n'] > 0, columns['raw'], base)
            features[tag+'_wmax'] = columns['wmax']
        return pd.DataFrame(features)

def implicit_scores(interactions, query, factor_sizes=(32, 128)):
    """Implicit-feedback ALS affinity: how likely the user is to read the book at all."""
    U = pd.Index(interactions.user.unique()); I = pd.Index(interactions.item.unique())
    matrix = sp.csr_matrix((np.ones(len(interactions), np.float32), (U.get_indexer(interactions.user), I.get_indexer(interactions.item))), shape=(len(U), len(I)))
    matrix.data[:] = 1
    u, i = U.get_indexer(query.user), I.get_indexer(query.item)
    out = {}
    for factors in factor_sizes:
        model = implicit_als.AlternatingLeastSquares(factors=factors, regularization=.1, iterations=20, alpha=20, random_state=42)
        model.fit(matrix, show_progress=False)
        uf, itf = np.asarray(model.user_factors), np.asarray(model.item_factors)
        out[f'als{factors}'] = np.where((u >= 0) & (i >= 0), np.einsum('ij,ij->i', uf[np.maximum(u, 0)], itf[np.maximum(i, 0)]), 0)
    return pd.DataFrame(out)


@njit
def _svdpp_epoch(order, r_ptr, r_item, r_val, n_ptr, n_item, mu, bu, bi, P, Q, Y, lr, lr_b, reg, reg_b, reg_y, seed):
    np.random.seed(seed)
    k = P.shape[1]
    for u in order:
        lo, hi = r_ptr[u], r_ptr[u+1]
        if hi == lo:
            continue
        nlo, nhi = n_ptr[u], n_ptr[u+1]
        s = 1.0/np.sqrt(max(nhi-nlo, 1))
        z = np.zeros(k)
        for t in range(nlo, nhi):
            z += Y[n_item[t]]
        z *= s
        acc = np.zeros(k)
        idx = np.arange(lo, hi)
        np.random.shuffle(idx)
        for t in idx:
            i = r_item[t]
            pe = P[u] + z
            e = r_val[t] - (mu + bu[u] + bi[i] + np.dot(pe, Q[i]))
            bu[u] += lr_b*(e - reg_b*bu[u])
            bi[i] += lr_b*(e - reg_b*bi[i])
            qi = Q[i].copy()
            Q[i] += lr*(e*pe - reg*Q[i])
            P[u] += lr*(e*qi - reg*P[u])
            acc += e*qi
        g = s*acc
        for t in range(nlo, nhi):
            j = n_item[t]
            Y[j] += lr*(g - reg_y*Y[j])

class FullSVDpp:
    """SVD++ (numba SGD) whose implicit set N(u) is every known interaction of the
    user: rated books, 0-star shelvings and the unlabeled query pairs."""
    def __init__(self, factors=100, epochs=30, lr=.007, lr_b=.007, reg=.05, reg_b=.01, reg_y=.05, decay=.93, seed=42):
        self.factors, self.epochs, self.lr, self.lr_b, self.reg, self.reg_b, self.reg_y, self.decay, self.seed = factors, epochs, lr, lr_b, reg, reg_b, reg_y, decay, seed

    def fit(self, rated, implicit):
        self.U = pd.Index(pd.concat([rated.user, implicit.user]).unique())
        self.I = pd.Index(pd.concat([rated.item, implicit.item]).unique())
        nu, ni = len(self.U), len(self.I)
        R = sp.csr_matrix((rated.rating.to_numpy(float), (self.U.get_indexer(rated.user), self.I.get_indexer(rated.item))), shape=(nu, ni))
        N = sp.csr_matrix((np.ones(len(implicit)), (self.U.get_indexer(implicit.user), self.I.get_indexer(implicit.item))), shape=(nu, ni))
        N.sum_duplicates(); N.data[:] = 1
        rng = np.random.default_rng(self.seed); k = self.factors
        self.mu = float(rated.rating.mean()); self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        self.P = rng.normal(0, .05, (nu, k)); self.Q = rng.normal(0, .05, (ni, k)); self.Y = np.zeros((ni, k))
        lr, lr_b = self.lr, self.lr_b
        for _ in range(self.epochs):
            _svdpp_epoch(rng.permutation(nu).astype(np.int64), R.indptr.astype(np.int64), R.indices.astype(np.int64), R.data,
                         N.indptr.astype(np.int64), N.indices.astype(np.int64), self.mu, self.bu, self.bi, self.P, self.Q, self.Y,
                         lr, lr_b, self.reg, self.reg_b, self.reg_y, int(rng.integers(1 << 30)))
            lr *= self.decay; lr_b *= self.decay
        s = 1/np.sqrt(np.maximum(np.asarray(N.sum(1)).ravel(), 1))
        self.Ueff = self.P + sp.diags(s) @ (N @ self.Y)
        return self

    def predict(self, users, items):
        u, i = self.U.get_indexer(users), self.I.get_indexer(items)
        okU, okI = u >= 0, i >= 0; uu, ii = np.maximum(u, 0), np.maximum(i, 0)
        p = self.mu + np.where(okU, self.bu[uu], 0) + np.where(okI, self.bi[ii], 0)
        p += np.where(okU & okI, np.einsum('ij,ij->i', self.Ueff[uu], self.Q[ii]), 0)
        return np.clip(p, 1, 5)

def distribution_features(rated, query):
    """Smoothed share of each star value (1-5) for the user and for the book."""
    out = {}
    prior = rated.rating.value_counts(normalize=True).sort_index()
    for entity, strength in (('user', 5.), ('item', 10.)):
        counts = pd.crosstab(rated[entity], rated.rating); n = counts.sum(1).to_numpy()[:, None]
        shares = (counts+strength*prior)/(n+strength)
        for star in shares.columns:
            out[f'{entity}_p{star}'] = query[entity].map(shares[star]).fillna(prior[star]).to_numpy()
    return pd.DataFrame(out)

def age_tenure_features(frame, query):
    """Task 3: the book's age and activity at the rating time, the user's tenure."""
    good = frame[(frame.timestamp > 9.5e8) & (frame.timestamp < 1.52e9)].sort_values('timestamp')
    t = query.timestamp.to_numpy(float); DAY = 86400.
    item_times = {k: g.to_numpy(float) for k, g in good.groupby('item').timestamp}
    user_times = {k: g.to_numpy(float) for k, g in good.groupby('user').timestamp}
    first = lambda times, key, q: (times.get(key, np.array([np.nan]))[int(q*(len(times.get(key, [0]))-1))])
    out = {'item_age_days': (t-np.array([first(item_times, i, .02) for i in query.item]))/DAY,
           'user_tenure_days': (t-np.array([first(user_times, u, .02) for u in query.user]))/DAY}
    for w in (45, 180):
        out[f'item_win{w}'] = np.log1p([np.searchsorted(a := item_times.get(i, np.array([])), x+w*DAY)-np.searchsorted(a, x-w*DAY) for i, x in zip(query.item, t)])
    out['user_frac_before'] = [np.searchsorted(a := user_times.get(u, np.array([0.])), x)/max(len(a), 1) for u, x in zip(query.user, t)]
    out['item_frac_before'] = [np.searchsorted(a := item_times.get(i, np.array([0.])), x)/max(len(a), 1) for i, x in zip(query.item, t)]
    return pd.DataFrame(out)

class BayesianSVDppFlipped:
    """Bayesian factorization machine (Gibbs sampling, myFM), 'Bayesian SVD++ flipped'
    (Rendle, Zhang & Koren 2019): user/book one-hot + normalized implicit sets
    (books the user interacted with; users who interacted with the book)."""
    def __init__(self, rank=48, n_iter=200, n_kept=150, seed=42):
        self.rank, self.n_iter, self.n_kept, self.seed = rank, n_iter, n_kept, seed
    def _blocks(self, users, items):
        from myfm import RelationBlock
        u = self.U.get_indexer(users); i = self.I.get_indexer(items)
        u = np.where(u >= 0, u, len(self.U)); i = np.where(i >= 0, i, len(self.I))
        return [RelationBlock(u, self.user_block), RelationBlock(i, self.item_block)]
    def fit(self, rated, implicit):
        import myfm
        self.U = pd.Index(pd.concat([rated.user, implicit.user]).unique())
        self.I = pd.Index(pd.concat([rated.item, implicit.item]).unique())
        nu, ni = len(self.U), len(self.I)
        N = sp.csr_matrix((np.ones(len(implicit)), (self.U.get_indexer(implicit.user), self.I.get_indexer(implicit.item))), shape=(nu, ni))
        N.sum_duplicates(); N.data[:] = 1
        Nu = sp.diags(1/np.sqrt(np.maximum(np.asarray(N.sum(1)).ravel(), 1)))@N
        NT = N.T.tocsr(); Ni = sp.diags(1/np.sqrt(np.maximum(np.asarray(NT.sum(1)).ravel(), 1)))@NT
        self.user_block = sp.vstack([sp.hstack([sp.identity(nu, format='csr'), Nu]), sp.csr_matrix((1, nu+ni))]).tocsr()
        self.item_block = sp.vstack([sp.hstack([sp.identity(ni, format='csr'), Ni]), sp.csr_matrix((1, ni+nu))]).tocsr()
        self.model = myfm.MyFMRegressor(rank=self.rank, random_seed=self.seed)
        self.model.fit(None, rated.rating.to_numpy(float), X_rel=self._blocks(rated.user, rated.item),
                       n_iter=self.n_iter, n_kept_samples=self.n_kept, callback=None)
        return self
    def predict(self, users, items):
        return np.clip(self.model.predict(None, self._blocks(users, items)), 1, 5)

RATING_SVD_CONFIGS = [
    dict(name='svd_1', factors=1, epochs=40, reg=.1, lr=.005, scale=(1, 5)),
    dict(name='svd_20', factors=20, epochs=60, reg=.1, lr=.005, scale=(1, 5)),
    dict(name='svd_100', factors=100, epochs=80, reg=.1, lr=.004, scale=(1, 5)),
    dict(name='svdpp_10', kind='svdpp', factors=10, epochs=30, reg=.05, lr=.005, scale=(1, 5)),
]
RATING_OLD_NEIGHBORS = {'jac100': ('jaccard', 100, 20.), 'cos100': ('cosine', 100, 20.),
                        'jac400': ('jaccard', 400, 50.), 'res200': ('residual', 200, 30.)}

def rating_features(frame, query, interactions, temporal=False):
    """Fit every base model on non-zero ratings of `frame` and featurize `query`.
    `interactions` (user,item pairs) may include unlabeled query pairs: they reveal
    *who read what* without revealing any rating."""
    rated = frame[frame.rating > 0]
    columns = {}
    for config in RATING_SVD_CONFIGS:
        print('  fitting', config['name'], flush=True)
        columns[config['name']] = predict_svd(fit_svd(config, rated), query)
    print('  fitting full-implicit SVD++', flush=True)
    columns['svdpp_full'] = FullSVDpp().fit(rated, interactions).predict(query.user, query.item)
    if not temporal:   # Task 1: Bayesian factorization machine (Gibbs sampling)
        print('  fitting Bayesian FM (myFM)', flush=True)
        columns['bayes_fm'] = BayesianSVDppFlipped().fit(rated, interactions).predict(query.user, query.item)
    print('  fitting residual neighborhoods', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        neighborhood = ResidualNeighborhood1().fit(rated)
        for name, (metric, k, shrink) in RATING_OLD_NEIGHBORS.items():
            columns[name] = neighborhood.predict(query, metric, k, shrink)
    parts = [pd.DataFrame(columns),
             pd.DataFrame(history_features(query, fit_history(rated))).add_prefix('hist_')]
    print('  fitting item/user kNN and implicit ALS', flush=True)
    parts.append(KNNFeatures().fit(rated, interactions).transform(query))
    parts.append(implicit_scores(interactions, query))
    parts.append(distribution_features(rated, query))
    if temporal:
        parts.append(age_tenure_features(frame, query))
        print('  fitting timeSVD++', flush=True)
        parts.append(pd.DataFrame({'timesvdpp': TimeSVDpp(use_day=True, reg_day=2., reg_bin=.5).fit(rated, interactions).predict(query.user, query.item, query.timestamp)}))
        print('  building temporal features', flush=True)
        _, time_basic = Task3Features().fit(rated).transform(query)
        strict, reconstruction = EnhancedTimeFeatures3().fit(rated).transform(query)
        with contextlib.redirect_stdout(io.StringIO()):
            targeted = TargetedFeatures3().fit(rated, neighborhood).transform(query)
        parts.append(pd.DataFrame(np.column_stack([time_basic, strict, reconstruction, targeted])).add_prefix('time_'))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

class SeedAveragedRegressor:
    """Average of gradient-boosted stackers with different seeds (steadier predictions)."""
    def __init__(self, seeds=(1, 2, 3)):
        self.seeds = seeds
    def fit(self, X, y):
        self.models = [HistGradientBoostingRegressor(max_iter=800, learning_rate=.03, max_leaf_nodes=63, min_samples_leaf=200,
                                                     l2_regularization=10, random_state=s).fit(X, y) for s in self.seeds]
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

def make_rating_stacker():
    return SeedAveragedRegressor()

def validate_rating_task(data, temporal, seed=42):
    """85/15 split; base models see only the 85%; the stacker is cross-validated on
    the non-zero 15% (out-of-fold MSE), then refit on all of it."""
    train, holdout = train_test_split(data, test_size=.15, random_state=seed)
    query = holdout[holdout.rating > 0].reset_index(drop=True)
    interactions = pd.concat([train[['user', 'item']], holdout[['user', 'item']]])
    X = rating_features(train, query, interactions, temporal)
    y = query.rating.to_numpy(float)
    oof = np.zeros(len(y))
    for fit_rows, score_rows in KFold(4, shuffle=True, random_state=0).split(X):
        oof[score_rows] = make_rating_stacker().fit(X.iloc[fit_rows], y[fit_rows]).predict(X.iloc[score_rows])
    report = pd.Series({'baseline svd_1 (non-zero)': mean_squared_error(y, np.clip(X.svd_1, 1, 5)),
                        'best single kNN': min(mean_squared_error(y, np.clip(X[c], 1, 5)) for c in X if c.endswith('_pred')),
                        'stacked (out-of-fold)': mean_squared_error(y, np.clip(oof, 1, 5))})
    return make_rating_stacker().fit(X, y), list(X.columns), report

def predict_rating_task(data, test, stacker, columns, temporal):
    interactions = pd.concat([data[['user', 'item']], test[['user', 'item']]])
    X = rating_features(data, test.reset_index(drop=True), interactions, temporal)
    assert list(X.columns) == columns, 'feature schema mismatch'
    return np.clip(stacker.predict(X), 1, 5)

# ---- timeSVD++ (Koren 2009) with the full implicit set: used by Task 3 ----
DAY = 86400.
T_LO, T_HI, BIN_DAYS = 9.5e8, 1.52e9, 180

@njit
def _tsvdpp_epoch(order, r_ptr, r_item, r_val, r_dev, r_bin, r_day, n_ptr, n_item,
           mu, bu, au, bud, bi, bib, P, AP, Q, Y, lr, lr_b, reg, reg_b, reg_a, reg_bin, reg_day, reg_y, seed):
    np.random.seed(seed)
    k = P.shape[1]
    for u in order:
        lo, hi = r_ptr[u], r_ptr[u+1]
        if hi == lo:
            continue
        nlo, nhi = n_ptr[u], n_ptr[u+1]
        s = 1.0/np.sqrt(max(nhi-nlo, 1))
        z = np.zeros(k)
        for t in range(nlo, nhi):
            z += Y[n_item[t]]
        z *= s
        acc = np.zeros(k)
        idx = np.arange(lo, hi)
        np.random.shuffle(idx)
        for t in idx:
            i = r_item[t]; dv = r_dev[t]; b = r_bin[t]; d = r_day[t]
            pe = P[u] + AP[u]*dv + z
            pred = mu + bu[u] + au[u]*dv + bi[i] + bib[i, b] + np.dot(pe, Q[i])
            if d >= 0:
                pred += bud[d]
            e = r_val[t] - pred
            bu[u] += lr_b*(e - reg_b*bu[u])
            au[u] += lr_b*.01*(e*dv - reg_a*au[u])
            bi[i] += lr_b*(e - reg_b*bi[i])
            bib[i, b] += lr_b*(e - reg_bin*bib[i, b])
            if d >= 0:
                bud[d] += lr_b*(e - reg_day*bud[d])
            qi = Q[i].copy()
            Q[i] += lr*(e*pe - reg*Q[i])
            P[u] += lr*(e*qi - reg*P[u])
            AP[u] += lr*.01*(e*qi*dv - reg_a*AP[u])
            acc += e*qi
        g = s*acc
        for t in range(nlo, nhi):
            j = n_item[t]
            Y[j] += lr*(g - reg_y*Y[j])

class TimeSVDpp:
    def __init__(self, factors=100, epochs=30, lr=.007, lr_b=.007, reg=.05, reg_b=.01, reg_a=1., reg_bin=.1, reg_day=.5,
                 reg_y=.05, decay=.93, use_day=True, seed=42):
        self.factors, self.epochs, self.lr, self.lr_b = factors, epochs, lr, lr_b
        self.reg, self.reg_b, self.reg_a, self.reg_bin, self.reg_day, self.reg_y = reg, reg_b, reg_a, reg_bin, reg_day, reg_y
        self.decay, self.use_day, self.seed = decay, use_day, seed

    def _time_columns(self, users, ts):
        t = np.clip(np.asarray(ts, float), T_LO, T_HI)
        u = self.U.get_indexer(users)
        mean_t = np.where(u >= 0, self.user_mean_t[np.maximum(u, 0)], t)
        diff = (t-mean_t)/DAY
        dev = np.sign(diff)*np.abs(diff)**.4
        b = ((t-T_LO)//(BIN_DAYS*DAY)).astype(np.int64)
        day = (t//DAY).astype(np.int64)
        return u, dev, b, day

    def fit(self, rated, implicit):
        self.U = pd.Index(pd.concat([rated.user, implicit.user]).unique())
        self.I = pd.Index(pd.concat([rated.item, implicit.item]).unique())
        nu, ni = len(self.U), len(self.I)
        u_all = self.U.get_indexer(rated.user)
        t = np.clip(rated.timestamp.to_numpy(float), T_LO, T_HI)
        self.user_mean_t = np.bincount(u_all, t, nu)/np.maximum(np.bincount(u_all, minlength=nu), 1)
        u, dev, b, day = self._time_columns(rated.user, rated.timestamp)
        i = self.I.get_indexer(rated.item)
        codes, uniques = pd.factorize(u.astype(np.int64)*30000+(day-10000))
        self.day_index = pd.Index(uniques)
        order = np.lexsort((i, u))
        u, i, dev, b, codes = u[order], i[order], dev[order], b[order], codes[order]
        r = rated.rating.to_numpy(float)[order]
        r_ptr = np.zeros(nu+1, np.int64); np.add.at(r_ptr, u+1, 1); r_ptr = np.cumsum(r_ptr)
        N = sp.csr_matrix((np.ones(len(implicit)), (self.U.get_indexer(implicit.user), self.I.get_indexer(implicit.item))), shape=(nu, ni))
        N.sum_duplicates(); N.data[:] = 1
        nbins = int((T_HI-T_LO)//(BIN_DAYS*DAY))+1
        rng = np.random.default_rng(self.seed); k = self.factors
        self.mu = float(r.mean())
        self.bu, self.au, self.bi = np.zeros(nu), np.zeros(nu), np.zeros(ni)
        self.bud = np.zeros(len(uniques)); self.bib = np.zeros((ni, nbins))
        self.P = rng.normal(0, .05, (nu, k)); self.AP = np.zeros((nu, k)); self.Q = rng.normal(0, .05, (ni, k)); self.Y = np.zeros((ni, k))
        day_arg = codes.astype(np.int64) if self.use_day else np.full(len(codes), -1, np.int64)
        lr, lr_b = self.lr, self.lr_b
        for _ in range(self.epochs):
            _tsvdpp_epoch(rng.permutation(nu).astype(np.int64), r_ptr, i.astype(np.int64), r, dev, b, day_arg,
                   N.indptr.astype(np.int64), N.indices.astype(np.int64), self.mu, self.bu, self.au, self.bud, self.bi, self.bib,
                   self.P, self.AP, self.Q, self.Y, lr, lr_b, self.reg, self.reg_b, self.reg_a, self.reg_bin, self.reg_day,
                   self.reg_y, int(rng.integers(1 << 30)))
            lr *= self.decay; lr_b *= self.decay
        s = 1/np.sqrt(np.maximum(np.asarray(N.sum(1)).ravel(), 1))
        self.Z = sp.diags(s) @ (N @ self.Y)
        return self

    def predict(self, users, items, ts):
        u, dev, b, day = self._time_columns(users, ts)
        i = self.I.get_indexer(items)
        okU, okI = u >= 0, i >= 0; uu, ii = np.maximum(u, 0), np.maximum(i, 0)
        b = np.clip(b, 0, self.bib.shape[1]-1)
        p = self.mu + np.where(okU, self.bu[uu]+self.au[uu]*dev, 0) + np.where(okI, self.bi[ii]+self.bib[ii, b], 0)
        if self.use_day:
            d = np.where(u >= 0, self.day_index.get_indexer(u.astype(np.int64)*30000+(day-10000)), -1)
            p += np.where(d >= 0, self.bud[np.maximum(d, 0)], 0)
        pe = self.P[uu]+self.AP[uu]*dev[:, None]+self.Z[uu]
        p += np.where(okU & okI, np.einsum('ij,ij->i', pe, self.Q[ii]), 0)
        return np.clip(p, 1, 5)
