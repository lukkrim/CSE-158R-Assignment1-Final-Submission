"""Pairwise interaction prediction (Tasks 2, 4 and 5): vectorized features + GBM.

Each candidate book gets features (item/user kNN affinity, RP3beta, implicit ALS/BPR,
popularity, time overlap); a symmetric gradient-boosted classifier on
[item1, item2, difference, user context] picks the book the user read. A candidate
already in the user's history is always the negative (users never re-read a book).
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from implicit import als as implicit_als, bpr as implicit_bpr
from sklearn.ensemble import HistGradientBoostingClassifier

DAY, YEAR = 86400., 365.25*86400

def topk_graph(B, k, shrink, metric='jaccard', chunk=512):
    B = B.tocsr().astype(np.float32); n = B.shape[0]
    count = np.asarray(B.sum(1)).ravel(); BT = B.T.tocsr(); R, C, V = [], [], []
    for start in range(0, n, chunk):
        O = (B[start:start+chunk] @ BT).tocsr()
        for o in range(O.shape[0]):
            i = start+o; lo, hi = O.indptr[o], O.indptr[o+1]
            js, c = O.indices[lo:hi], O.data[lo:hi]; keep = js != i; js, c = js[keep], c[keep]
            if not len(js): continue
            den = count[i]+count[js]-c if metric == 'jaccard' else np.sqrt(count[i]*count[js])
            w = c/np.maximum(den, 1)*c/(c+shrink)
            if len(w) > k: t = np.argpartition(w, -k)[-k:]; js, w = js[t], w[t]
            R.append(np.full(len(js), i)); C.append(js); V.append(w)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))

def rp3beta_graph(M, beta=.5, k=300, chunk=256):
    """Random-walk item graph with popularity penalty (RP3beta)."""
    ulen = np.asarray(M.sum(1)).ravel(); pop = np.asarray(M.sum(0)).ravel()
    Pui = (sp.diags(1/np.maximum(ulen, 1))@M).tocsr(); Piu = (sp.diags(1/np.maximum(pop, 1))@M.T).tocsr()
    pen = 1/np.maximum(pop, 1)**beta; n = M.shape[1]; R, C, V = [], [], []
    for s in range(0, n, chunk):
        W = (Piu[s:s+chunk]@Pui).toarray()*pen[None, :]
        for o in range(W.shape[0]):
            w = W[o]; w[s+o] = 0; t = np.argpartition(w, -k)[-k:]; t = t[w[t] > 0]
            R.append(np.full(len(t), s+o)); C.append(t); V.append(w[t])
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))

def row_max(W):
    if not len(W.data): return np.zeros(W.shape[0])
    m = np.maximum.reduceat(np.append(W.data, 0), np.minimum(W.indptr[:-1], len(W.data)))
    return np.where(np.diff(W.indptr) > 0, m, 0)

class PairwiseFeatures:
    def __init__(self, temporal, seed=42):
        self.temporal, self.seed = temporal, seed

    def fit(self, frame):
        self.U = pd.Index(frame.user.unique()); self.I = pd.Index(frame.item.unique())
        u, i = self.U.get_indexer(frame.user), self.I.get_indexer(frame.item)
        M = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(len(self.U), len(self.I))); M.data[:] = 1
        self.M, self.MT = M, M.T.tocsr()
        self.pop = np.asarray(M.sum(0)).ravel(); self.ulen = np.asarray(M.sum(1)).ravel()
        r = frame.rating.to_numpy(float); nz = r > 0
        self.item_mean = (np.bincount(i[nz], r[nz], len(self.I))+20*r[nz].mean())/(np.bincount(i[nz], minlength=len(self.I))+20)
        self.graphs = {f'i{k}': topk_graph(self.MT, k, 10.) for k in (50, 300, 2000)}
        self.graphs['icos300'] = topk_graph(self.MT, 300, 10., 'cosine')
        self.graphs['rp3'] = rp3beta_graph(M)
        self.ugraphs = {f'u{k}': topk_graph(M, k, 10.) for k in (50, 300)}
        self.factors = {}
        for f in (64, 128):
            m = implicit_als.AlternatingLeastSquares(factors=f, regularization=.1, iterations=25, alpha=20, random_state=self.seed); m.fit(M, show_progress=False)
            self.factors[f'als{f}'] = (np.asarray(m.user_factors), np.asarray(m.item_factors))
        for f, it in ((64, 150), (128, 300)):
            m = implicit_bpr.BayesianPersonalizedRanking(factors=f, regularization=.01, iterations=it, learning_rate=.05, random_state=self.seed); m.fit(M, show_progress=False)
            self.factors[f'bpr{f}'] = (np.asarray(m.user_factors), np.asarray(m.item_factors))
        # timestamps (corrupt values outside 2000-2020 are ignored for histograms/quantiles)
        t = frame.timestamp.to_numpy(float); good = (t > 9.5e8) & (t < 1.6e9)
        self.lo, self.w = 9.5e8, 91*DAY; nb = int((1.6e9-self.lo)//self.w)+1
        b = ((t[good]-self.lo)//self.w).astype(int)
        def hist(rows, n):
            H = sp.csr_matrix((np.ones(len(rows)), (rows, b)), shape=(n, nb)); s = np.asarray(H.sum(1)).ravel()
            return (sp.diags(1/np.maximum(s, 1))@H).tocsr()
        self.UH = hist(u[good], len(self.U)); IH = hist(i[good], len(self.I))
        K = sum(np.eye(nb, k=d)*w for d, w in ((0, .5), (1, .2), (-1, .2), (2, .05), (-2, .05)))
        self.IH = (IH@sp.csr_matrix(K)).tocsr()
        g = pd.DataFrame({'u': u[good], 'i': i[good], 't': t[good]})
        self.uq = {q: g.groupby('u').t.quantile(q).reindex(range(len(self.U))).to_numpy() for q in (.02, .5, .98)}
        self.iq = {q: g.groupby('i').t.quantile(q).reindex(range(len(self.I))).to_numpy() for q in (.02, .5, .98)}
        if self.temporal:
            tc = np.clip(t, 0, 2e9)
            o = np.lexsort((tc, i)); self.ikeys = i[o]*4e9+tc[o]
            o = np.lexsort((tc, u)); self.ukeys = u[o]*4e9+tc[o]
            self.T = sp.csr_matrix((tc+1., (u, i)), shape=M.shape)
        return self

    def item_side(self, users, items, ts=None, chunk=10000):
        parts = [self._side(users[s:s+chunk], items[s:s+chunk], None if ts is None else ts[s:s+chunk]) for s in range(0, len(users), chunk)]
        return pd.concat(parts, ignore_index=True)

    def _side(self, users, items, ts):
        u, i = self.U.get_indexer(users), self.I.get_indexer(items)
        ok = (u >= 0) & (i >= 0); uu, ii = np.maximum(u, 0), np.maximum(i, 0)
        z = lambda v: np.where(ok, v, 0.)
        F = {'logpop': np.log1p(np.where(i >= 0, self.pop[ii], 0)), 'item_mean': np.where(i >= 0, self.item_mean[ii], 0),
             'seen': z(np.asarray(self.M[uu, ii]).ravel())}
        H = self.M[uu]
        for name, G in self.graphs.items():
            W = G[ii].multiply(H).tocsr(); s = np.asarray(W.sum(1)).ravel(); n = np.diff(W.indptr)
            F[name+'_sum'] = z(np.log1p(100*s)); F[name+'_max'] = z(row_max(W))
            F[name+'_frac'] = z(n/np.maximum(self.ulen[uu], 1)); F[name+'_mean'] = z(s/np.maximum(self.ulen[uu], 1))
        R = self.MT[ii]
        for name, G in self.ugraphs.items():
            W = G[uu].multiply(R).tocsr()
            F[name+'_sum'] = z(np.log1p(100*np.asarray(W.sum(1)).ravel())); F[name+'_max'] = z(row_max(W)); F[name+'_n'] = z(np.log1p(np.diff(W.indptr)))
        for name, (P, Q) in self.factors.items():
            F[name] = z(np.einsum('ij,ij->i', P[uu], Q[ii]))
        F['t_overlap'] = z(np.asarray(self.UH[uu].multiply(self.IH[ii]).sum(1)).ravel())
        F['item_first_minus_user_last'] = (self.iq[.02][ii]-self.uq[.98][uu])/YEAR
        F['user_first_minus_item_last'] = (self.uq[.02][uu]-self.iq[.98][ii])/YEAR
        F['item_med_minus_user_med'] = (self.iq[.5][ii]-self.uq[.5][uu])/YEAR
        if self.temporal:
            t = np.clip(np.asarray(ts, float), 0, 2e9)
            F['t_minus_item_first'] = (t-self.iq[.02][ii])/YEAR; F['item_last_minus_t'] = (self.iq[.98][ii]-t)/YEAR
            F['t_minus_item_med'] = (t-self.iq[.5][ii])/YEAR
            count = lambda keys, idx, w: np.searchsorted(keys, idx*4e9+t+w, 'right')-np.searchsorted(keys, idx*4e9+t-w, 'left')
            for d in (15, 45, 180, 365):
                c = count(self.ikeys, ii, d*DAY); F[f'iwin{d}'] = np.log1p(c); F[f'ishare{d}'] = c/(self.pop[ii]+5)
            for d in (30, 180): F[f'uwin{d}'] = np.log1p(count(self.ukeys, uu, d*DAY))
            HT = self.T[uu]
            for name in ('i300', 'icos300', 'rp3'):
                G = self.graphs[name]; W = G[ii].multiply(H).tocsr(); WT = G[ii].multiply(HT).tocsr()
                rows = np.repeat(np.arange(len(ii)), np.diff(W.indptr))
                dt = np.abs(WT.data/np.maximum(W.data, 1e-12)-1.-t[rows])/DAY
                for h in (30, 180, 730):
                    wd = W.data*np.exp2(-dt/h)
                    F[f'{name}_dec{h}'] = z(np.log1p(100*np.bincount(rows, wd, len(ii))))
                    F[f'{name}_decmax{h}'] = z(np.maximum.reduceat(np.append(wd, 0), np.minimum(W.indptr[:-1], len(wd)))*(np.diff(W.indptr) > 0))
        return pd.DataFrame({k: np.nan_to_num(np.asarray(v, float)) for k, v in F.items()})

    def pair_features(self, pairs):
        ts = pairs.timestamp.to_numpy() if self.temporal else None
        users = pairs.user.to_numpy()
        a = self.item_side(users, pairs.item1.to_numpy(), ts); b = self.item_side(users, pairs.item2.to_numpy(), ts)
        uu = np.maximum(self.U.get_indexer(users), 0)
        ctx = pd.DataFrame({'user_len': np.log1p(self.ulen[uu])})
        return a, b, ctx

def design(a, b, ctx, swap=False):
    A, B = (b, a) if swap else (a, b)
    return np.column_stack([A.to_numpy(), B.to_numpy(), A.to_numpy()-B.to_numpy(), ctx.to_numpy()])

def sample_pairs(holdout, train_items, seed):
    """Negative = the item of a random training interaction (as the instructor sampled them)."""
    rng = np.random.default_rng(seed); pos = holdout.item.to_numpy()
    neg = train_items[rng.integers(len(train_items), size=len(pos))]
    while (same := neg == pos).any(): neg[same] = train_items[rng.integers(len(train_items), size=same.sum())]
    first = rng.integers(2, size=len(pos)).astype(bool)
    out = pd.DataFrame({'user': holdout.user.to_numpy(), 'item1': np.where(first, pos, neg), 'item2': np.where(first, neg, pos), 'label': first.astype(int)})
    if 'timestamp' in holdout: out['timestamp'] = holdout.timestamp.to_numpy()
    return out

def build_training_pairs(data, temporal, rounds=5, rows_per_round=30000, seed=42):
    """Each round holds out a small random slice, fits all base models on the rest,
    and featurizes positive/negative pairs for the held-out interactions."""
    rng = np.random.default_rng(seed); order = rng.permutation(len(data)); blocks = []
    for r in range(rounds):
        held = order[r*rows_per_round:(r+1)*rows_per_round]
        train = data.drop(index=data.index[held]); holdout = data.iloc[held]
        print(f'  round {r+1}/{rounds}: fitting base models', flush=True)
        pf = PairwiseFeatures(temporal).fit(train)
        pairs = sample_pairs(holdout, train.item.to_numpy(), seed+r)
        blocks.append((pairs, *pf.pair_features(pairs)))
    pairs = pd.concat([b[0] for b in blocks], ignore_index=True)
    return pairs, *(pd.concat([b[k] for b in blocks], ignore_index=True) for k in (1, 2, 3))

def make_pair_classifier():
    return HistGradientBoostingClassifier(max_iter=1000, learning_rate=.04, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=10, random_state=0)

def fit_pair_classifier(a, b, ctx, y):
    return make_pair_classifier().fit(np.vstack([design(a, b, ctx), design(a, b, ctx, True)]), np.concatenate([y, 1-y]))

def pair_probability(model, a, b, ctx):
    """Symmetric P(item1 is positive), with the seen-item rule applied."""
    p = .5*(model.predict_proba(design(a, b, ctx))[:, 1]+1-model.predict_proba(design(a, b, ctx, True))[:, 1])
    s1, s2 = a.seen.to_numpy() > 0, b.seen.to_numpy() > 0
    return np.where(s1 & ~s2, 0., np.where(s2 & ~s1, 1., p))
