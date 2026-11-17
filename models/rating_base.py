"""Base rating models shared by Tasks 1 and 3.

* fit_svd / predict_svd: Surprise SVD, SVD++ and baseline models from a config dict.
* fit_history / history_features: user/book rating statistics (overall and recent).
* ResidualNeighborhood1: item neighborhoods on rating residuals around fitted biases.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from surprise import BaselineOnly, Dataset, Reader, SVD, SVDpp

SEED = 42
RECENT_N = 20  # window for recent-rating history features


def fit_svd(config, fit_frame):
    reader = Reader(rating_scale=config["scale"])
    data = Dataset.load_from_df(fit_frame[["user", "item", "rating"]], reader)
    model = SVD(
        n_factors=config.get("factors", 1), n_epochs=config.get("epochs", 25),
        reg_all=config.get("reg", .02), lr_all=config.get("lr", .005),
        biased=True, random_state=config.get("seed", 42),
    )
    if config.get("kind") == "svdpp":
        model = SVDpp(n_factors=config.get("factors", 1), n_epochs=config.get("epochs", 25), reg_all=config.get("reg", .02), lr_all=config.get("lr", .005), random_state=config.get("seed", 42), cache_ratings=True)
    if config.get("kind") == "bias":
        model = BaselineOnly(bsl_options=dict(method="als", n_epochs=25, reg_u=10, reg_i=15))
    return model.fit(data.build_full_trainset())

def predict_svd(model, frame):
    values = np.fromiter(
        (model.predict(u, i).est for u, i in zip(frame["user"], frame["item"])),
        dtype=float, count=len(frame),
    )
    return np.clip(values, 0, 5)


def fit_history(frame, recent_n=RECENT_N):
    mean = frame["rating"].mean()
    users = frame.groupby("user")["rating"].agg(["sum", "count"])
    items = frame.groupby("item")["rating"].agg(["sum", "count"])
    ordered = frame.sort_values("timestamp")
    stats = {
        "mean": mean,
        "ub": (users["sum"] - users["count"] * mean) / (users["count"] + 10),
        "ib": (items["sum"] - items["count"] * mean) / (items["count"] + 20),
        "uc": users["count"], "ic": items["count"],
        "ur": ordered.groupby("user", sort=False).tail(recent_n).groupby("user")["rating"].mean(),
        "ir": ordered.groupby("item", sort=False).tail(recent_n).groupby("item")["rating"].mean(),
    }
    # Smoothed proportions use only the base-model training rows.
    stats['distribution'] = {}
    indicators = {'zero': frame.rating.eq(0), 'low': frame.rating.le(2), 'five': frame.rating.eq(5)}
    for entity, strength in (('user', 20.), ('item', 40.)):
        groups = frame.groupby(entity).rating
        count = groups.size()
        stats['distribution'][entity + '_variance'] = (groups.var().fillna(0)*count + strength*frame.rating.var())/(count+strength)
        for label, indicator in indicators.items():
            successes = indicator.groupby(frame[entity]).sum()
            stats['distribution'][entity + '_' + label] = (successes+strength*indicator.mean())/(count+strength)
    stats['distribution_defaults'] = {key: float(frame.rating.var()) if key.endswith('variance') else float(indicators[key.split('_')[-1]].mean()) for key in stats['distribution']}
    return stats


def history_features(frame, stats):
    m = stats["mean"]
    mapped = lambda key, default: frame[key[0]].map(stats[key[1]]).fillna(default).to_numpy(float)
    ub, ib = mapped(("user", "ub"), 0), mapped(("item", "ib"), 0)
    basic = np.column_stack([
        np.clip(m + ub + ib, 0, 5),
        mapped(("user", "ur"), m), mapped(("item", "ir"), m),
        np.log1p(mapped(("user", "uc"), 0)), np.log1p(mapped(("item", "ic"), 0)),
    ])
    distribution = np.column_stack([frame[key.split('_')[0]].map(series).fillna(stats['distribution_defaults'][key]).to_numpy(float) for key,series in stats['distribution'].items()])
    return np.column_stack([basic, distribution])


class ResidualNeighborhood1:
    """Overlap-shrunk item similarities; predict residuals around fitted biases."""
    def fit(self, frame):
        self.users = {v:k for k,v in enumerate(frame.user.unique())}
        self.items = {v:k for k,v in enumerate(frame.item.unique())}
        u = frame.user.map(self.users).to_numpy()
        i = frame.item.map(self.items).to_numpy()
        r = frame.rating.to_numpy(float)
        self.mean = r.mean()
        self.ub, self.ib = np.zeros(len(self.users)), np.zeros(len(self.items))
        for _ in range(15):
            self.ub = np.bincount(u, weights=r-self.mean-self.ib[i], minlength=len(self.users))/(np.bincount(u, minlength=len(self.users))+10)
            self.ib = np.bincount(i, weights=r-self.mean-self.ub[u], minlength=len(self.items))/(np.bincount(i, minlength=len(self.items))+20)
        self.binary = sparse.csr_matrix((np.ones(len(frame)), (i,u)), shape=(len(self.items),len(self.users)))
        self.binary.data[:] = 1
        self.counts = np.asarray(self.binary.sum(axis=1)).ravel()
        self.history = {}
        residual = r-self.mean-self.ub[u]-self.ib[i]
        for user, group in pd.DataFrame(dict(user=u,item=i,residual=residual)).groupby('user'):
            self.history[user] = (group.item.to_numpy(), group.residual.to_numpy())
        self.residual_matrix = sparse.csr_matrix((residual,(i,u)),shape=self.binary.shape)
        self.residual_norm = np.sqrt(np.asarray(self.residual_matrix.multiply(self.residual_matrix).sum(axis=1)).ravel())
        self._neighbors = lru_cache(maxsize=10000)(self._compute_neighbors)
        return self

    def _compute_neighbors(self, item, metric):
        overlaps = (self.binary[item] @ self.binary.T).tocsr()
        js, common = overlaps.indices, overlaps.data
        keep = js != item
        js, common = js[keep], common[keep]
        denominator = self.counts[item]+self.counts[js]-common if metric == 'jaccard' else np.sqrt(self.counts[item]*self.counts[js])
        weights = common/np.maximum(denominator,1) * common/(common+20.)
        # Keep all co-read neighbors; choose top k among this user's rated items.
        return dict(zip(js,weights))

    def predict(self, frame, metric='cosine', k=100, shrink=20.):
        # Build a bounded item graph once, then score many rows together per user.
        if not hasattr(self, '_graphs'):
            self._graphs = {}
        key = (metric,k,shrink)
        if key not in self._graphs:
            rows, columns, values = [], [], []
            size = len(self.items)
            for start in range(0, size, 128):
                overlap = (self.binary[start:start+128] @ self.binary.T).tocsr()
                residual_product = (self.residual_matrix[start:start+128] @ self.residual_matrix.T).tocsr() if metric == 'residual' else None
                for offset in range(overlap.shape[0]):
                    item = start + offset
                    lo,hi = overlap.indptr[offset:offset+2]
                    js, common = overlap.indices[lo:hi], overlap.data[lo:hi]
                    denominator = self.counts[item]+self.counts[js]-common if metric == 'jaccard' else np.sqrt(self.counts[item]*self.counts[js])
                    if metric == 'residual':
                        numerator = residual_product[offset,js].toarray().ravel()
                        denominator = self.residual_norm[item]*self.residual_norm[js]
                        weights = numerator/np.maximum(denominator,1e-8)*common/(common+shrink)
                    else:
                        weights = common/np.maximum(denominator,1)*common/(common+shrink)
                    weights[js == item] = 0
                    top = np.argpartition(np.abs(weights), -min(k,len(weights)))[-k:] if len(weights) else np.array([],dtype=int)
                    rows.extend([item]*len(top)); columns.extend(js[top]); values.extend(weights[top])
                if start % 2048 == 0:
                    print(f'{metric}: graph {min(start+128,size):,}/{size:,} books',flush=True)
            self._graphs[key] = sparse.csr_matrix((values,(rows,columns)),shape=(size,size))
        graph = self._graphs[key]
        users = frame.user.map(self.users).fillna(-1).to_numpy(int)
        items = frame.item.map(self.items).fillna(-1).to_numpy(int)
        prediction = np.full(len(frame),self.mean)
        known_u, known_i = users >= 0, items >= 0
        prediction[known_u] += self.ub[users[known_u]]
        prediction[known_i] += self.ib[items[known_i]]
        # Group requested rows by user; sparse matrix-vector products replace
        # millions of Python dictionary lookups over entire rating histories.
        groups = pd.Series(np.arange(len(frame))).groupby(users,sort=False)
        for user, positions in groups:
            if user < 0:
                continue
            positions = positions.to_numpy()
            positions = positions[items[positions] >= 0]
            if not len(positions):
                continue
            history, residual = self.history[user]
            weights = graph[items[positions]][:,history]
            prediction[positions] += np.asarray(weights @ residual).ravel()/(np.asarray(abs(weights).sum(axis=1)).ravel()+.1)
        return np.clip(prediction,0,5)
