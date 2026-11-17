"""Task 5 additions: directed transitions from the user's most recent books."""
import numpy as np, pandas as pd, scipy.sparse as sp

from .pairwise import PairwiseFeatures, row_max

class SequentialFeatures(PairwiseFeatures):
    def fit(self, frame):
        super().fit(frame)                      # frame is ordered by user, then time
        u, i = self.U.get_indexer(frame.user), self.I.get_indexer(frame.item)
        n = len(self.I); rows, cols, vals = [], [], []
        same_user = lambda lag: u[lag:] == u[:-lag]
        for lag in (1, 2, 3):
            m = same_user(lag); rows.append(i[:-lag][m]); cols.append(i[lag:][m]); vals.append(np.full(m.sum(), 1./lag))
        Tr = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        out = np.asarray(Tr.sum(1)).ravel()
        self.TrT = Tr.T.tocsr()                                  # candidate -> predecessors
        self.TrnT = (sp.diags(1/np.maximum(out, 1))@Tr).T.tocsr()  # row-normalized P(next | prev)
        # last 10 books of every user, most recent first
        order = pd.DataFrame({'u': u, 'i': i}).groupby('u', sort=False).tail(10)
        self.recent = order.groupby('u').i.apply(lambda s: s.to_numpy()[::-1]).reindex(range(len(self.U))).to_dict()
        return self

    def _recent_matrix(self, uu, window, half):
        rows, cols, vals = [], [], []
        for r, user in enumerate(uu):
            seq = self.recent.get(user)
            if seq is None or not isinstance(seq, np.ndarray): continue
            seq = seq[:window]; rows.append(np.full(len(seq), r)); cols.append(seq); vals.append(np.exp2(-np.arange(len(seq))/half))
        if not rows: return sp.csr_matrix((len(uu), len(self.I)))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(uu), len(self.I)))

    def _side(self, users, items, ts):
        F = super()._side(users, items, ts)
        u, i = self.U.get_indexer(users), self.I.get_indexer(items)
        ok = (u >= 0) & (i >= 0); uu, ii = np.maximum(u, 0), np.maximum(i, 0)
        for window, half in ((1, 1.), (3, 1.5), (10, 3.)):
            L = self._recent_matrix(uu, window, half)
            for name, M in (('tr', self.TrT), ('trn', self.TrnT)):
                W = M[ii].multiply(L).tocsr()
                F[f'{name}{window}_sum'] = np.where(ok, np.log1p(100*np.asarray(W.sum(1)).ravel()), 0)
                F[f'{name}{window}_max'] = np.where(ok, row_max(W), 0)
            G = self.graphs['rp3']; W = G[ii].multiply(L).tocsr()
            F[f'rp3_last{window}'] = np.where(ok, np.log1p(1e4*np.asarray(W.sum(1)).ravel()), 0)
        # share of the book's reads after the user's last activity
        t = np.clip(np.asarray(ts, float), 0, 2e9)
        after = np.searchsorted(self.ikeys, ii*4e9+4e9-1, 'right')-np.searchsorted(self.ikeys, ii*4e9+t, 'right')
        F['item_after_n'] = np.log1p(after); F['item_after_share'] = after/(self.pop[ii]+2)
        return F
