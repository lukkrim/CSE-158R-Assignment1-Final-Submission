"""Task 3 timestamp features for the rating stack.

Time-binned user/book effects, rating trends, strictly-earlier and nearby-in-time
rating statistics, a joint temporal bias model, and time-decayed neighborhoods.
"""
import numpy as np
import pandas as pd

from .rating_base import ResidualNeighborhood1, fit_history, history_features


class Task3Features:
    def fit(self, frame):
        self.frame = frame
        self.stats = fit_history(frame)
        self.center = float(frame.timestamp.mean())
        self.global_mean = float(frame.rating.mean())
        self.groups = {}
        for entity in ('user','item'):
            self.groups[entity] = {key:(g.timestamp.to_numpy(float),g.rating.to_numpy(float)) for key,g in frame.sort_values('timestamp').groupby(entity,sort=False)}
        self.periods = {}
        for days in (90,365):
            bins = (frame.timestamp//(days*86400)).astype(int)
            grouped = frame.groupby(bins).rating.agg(['sum','count'])
            self.periods[days] = (grouped['sum']+50*self.global_mean)/(grouped['count']+50)
        return self

    def transform(self, frame):
        # Use order-independent history for the timestamp-free reference.
        basic = history_features(frame,self.stats).copy()
        basic[:,1] = frame.user.map(self.stats['ub']).fillna(0).to_numpy()+self.global_mean
        basic[:,2] = frame.item.map(self.stats['ib']).fillna(0).to_numpy()+self.global_mean
        extra = [(frame.timestamp.to_numpy(float)-self.center)/(365.25*86400)]
        for days in (90,365):
            bins=(frame.timestamp//(days*86400)).astype(int)
            extra.append(bins.map(self.periods[days]).fillna(self.global_mean).to_numpy())
        for entity in ('user','item'):
            rows=[]
            for key,t in zip(frame[entity],frame.timestamp):
                times,ratings=self.groups[entity].get(key,(np.array([]),np.array([])))
                # Strictly earlier observations for local history features.
                end=np.searchsorted(times,t,side='left')
                prior=ratings[:end]
                recent=prior[-20:]
                left=np.searchsorted(times,t-180*86400,side='left')
                local=ratings[left:end]
                smoothed=(recent.sum()+10*self.global_mean)/(len(recent)+10)
                local_mean=(local.sum()+20*self.global_mean)/(len(local)+20)
                gap=(t-times[end-1])/86400 if end else np.nan
                # Regularized linear trend from available earlier ratings.
                if end>1:
                    dt=(times[:end]-t)/(365.25*86400)
                    centered=dt-dt.mean()
                    slope=np.dot(centered,prior-prior.mean())/(np.dot(centered,centered)+20)
                    trend=prior.mean()-slope*dt.mean()
                else: trend=self.global_mean
                rows.append([smoothed,local_mean,np.log1p(end),np.log1p(len(local)),gap,trend])
            extra.extend(np.asarray(rows,float).T)
        return basic,np.column_stack(extra)

class EnhancedTimeFeatures3:
    def fit(self, frame):
        self.mean=float(frame.rating.mean())
        users=frame.groupby('user').rating.agg(['sum','count'])
        items=frame.groupby('item').rating.agg(['sum','count'])
        self.ub=(users['sum']-users['count']*self.mean)/(users['count']+10)
        self.ib=(items['sum']-items['count']*self.mean)/(items['count']+20)
        data=frame.copy()
        data['residual']=data.rating-self.mean-data.user.map(self.ub)-data.item.map(self.ib)
        self.priors=np.array([frame.rating.eq(0).mean(),frame.rating.le(2).mean(),frame.rating.eq(5).mean()])
        self.groups={};self.bin_bias={}
        for entity,regularization in [('user',20.),('item',40.)]:
            ordered=data.sort_values('timestamp')
            self.groups[entity]={key:(g.timestamp.to_numpy(float),g.rating.to_numpy(float),g.residual.to_numpy(float)) for key,g in ordered.groupby(entity,sort=False)}
            for days in (30,90,365):
                bins=(data.timestamp//(days*86400)).astype(int)
                groups=data.groupby([data[entity],bins]).residual.agg(['sum','count'])
                self.bin_bias[entity,days]=groups['sum']/(groups['count']+regularization)
        return self

    def transform(self, frame):
        columns=[];strict_columns=[]
        for entity in ('user','item'):
            for days in (30,90,365):
                index=pd.MultiIndex.from_arrays([frame[entity],(frame.timestamp//(days*86400)).astype(int)])
                columns.append(self.bin_bias[entity,days].reindex(index).fillna(0).to_numpy())
            rows=[];symmetric=[]
            for key,t in zip(frame[entity],frame.timestamp):
                times,rating,residual=self.groups[entity].get(key,(np.array([]),np.array([]),np.array([])))
                end=np.searchsorted(times,t,side='left')
                features=[];nearby=[]
                for days in (30,90,365):
                    left=np.searchsorted(times,t-days*86400,side='left')
                    right=np.searchsorted(times,t+days*86400,side='right')
                    r=rating[left:end]; e=residual[left:end]
                    proportions=(np.array([(r==0).sum(),(r<=2).sum(),(r==5).sum()])+20*self.priors)/(len(r)+20)
                    features.extend([e.sum()/(len(e)+20),*proportions,np.log1p(len(e))])
                    # Separate reconstruction feature family: training rows on either side.
                    nearby.append(residual[left:right].sum()/(right-left+20))
                rows.append(features);symmetric.append(nearby)
            strict_columns.extend(np.asarray(rows,float).T)
            columns.extend(np.asarray(symmetric,float).T)
        return np.column_stack(strict_columns),np.column_stack(columns)

def temporal_neighbors3(model, frame, training, metric='jaccard', k=200, shrink=20., half_life_days=365.):
    # Ensure the sparse graph exists; its ordinary predictions are also useful.
    ordinary=model.predict(frame,metric,k,shrink)
    graph=model._graphs[(metric,k,shrink)]
    histories={key:g for key,g in training.groupby('user',sort=False)}
    half_lives=np.array([30.,90.,180.,365.,730.])
    out=[]
    for user,item,t in zip(frame.user,frame.item,frame.timestamp):
        u,i=model.users.get(user),model.items.get(item)
        base=model.mean+(model.ub[u] if u is not None else 0)+(model.ib[i] if i is not None else 0)
        prediction=np.full(len(half_lives),base)
        if u is not None and i is not None:
            g=histories[user]
            history=g.item.map(model.items).to_numpy(int)
            weights=graph[i,history].toarray().ravel()
            # All are training observations; closeness is relative to target time.
            decay=np.exp2(-np.abs(g.timestamp.to_numpy(float)-t)[:,None]/(half_lives[None,:]*86400))
            weights=weights[:,None]*decay
            residual=g.rating.to_numpy(float)-model.mean-model.ub[u]-model.ib[history]
            prediction+=np.sum(weights*residual[:,None],axis=0)/(np.abs(weights).sum(axis=0)+.1)
        out.append(prediction)
    return ordinary,np.clip(out,0,5)

class JointTemporalBias3:
    """Joint regularized least-squares user/item and period effects."""
    def _matrix(self, frame):
        import scipy.sparse as sp
        rows=[];cols=[];values=[]
        for mapping,keys,offset,penalty in zip(self.maps,self._keys(frame),self.offsets,self.penalties):
            ids=pd.Series(keys).map(mapping).fillna(-1).to_numpy(int)
            valid=ids>=0
            rows.append(np.flatnonzero(valid));cols.append(ids[valid]+offset)
            values.append(np.full(valid.sum(),1/np.sqrt(penalty)))
        return sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(frame),self.width))

    def _keys(self, frame):
        keys=[frame.user.to_list(),frame.item.to_list()]
        for days in (90,365):
            bins=(frame.timestamp//(days*86400)).astype('int64').to_numpy()
            keys.extend([list(zip(frame.user,bins)),list(zip(frame.item,bins))])
        return keys

    def fit(self, frame):
        from scipy.sparse.linalg import lsqr
        self.mean=float(frame.rating.mean())
        self.maps=[];self.offsets=[];self.width=0
        self.penalties=[10.,20.,30.,50.,30.,50.]
        for keys in self._keys(frame):
            mapping={key:i for i,key in enumerate(dict.fromkeys(keys))}
            self.maps.append(mapping);self.offsets.append(self.width);self.width+=len(mapping)
        matrix=self._matrix(frame)
        solved=lsqr(matrix,frame.rating.to_numpy()-self.mean,damp=1.,atol=1e-5,btol=1e-5,iter_lim=150)
        self.coefficients=solved[0]
        print('Joint temporal bias iterations:',solved[2],'termination:',solved[1],flush=True)
        return self

    def predict(self, frame):
        return self.mean+np.asarray(self._matrix(frame)@self.coefficients).ravel()

class TargetedFeatures3:
    def fit(self, frame, neighborhood=None):
        self.bias=JointTemporalBias3().fit(frame)
        self.frame=frame
        self.neighbors=neighborhood if neighborhood is not None else ResidualNeighborhood1().fit(frame)
        self.residual=frame.rating.to_numpy()-self.bias.predict(frame)
        self.groups={}
        data=frame.copy();data['_residual']=self.residual
        for entity in ('user','item'):
            self.groups[entity]={key:(g.timestamp.to_numpy(float),g.rating.to_numpy(float),g._residual.to_numpy(float)) for key,g in data.sort_values('timestamp').groupby(entity,sort=False)}
        self.priors=np.array([frame.rating.eq(0).mean(),frame.rating.le(2).mean(),frame.rating.eq(5).mean()])
        self.user_rows={key:ix for key,ix in frame.reset_index(drop=True).groupby('user',sort=False).indices.items()}
        item_ids=frame.item.map(self.neighbors.items).to_numpy(int)
        timestamps=frame.timestamp.to_numpy(float)
        self.user_arrays={key:(item_ids[ix],timestamps[ix],self.residual[ix]) for key,ix in self.user_rows.items()}
        # Build one sparse graph, reused across both queries and decay scales.
        self.neighbors.predict(frame.iloc[:1],'jaccard',200,20.)
        return self

    def transform(self, frame):
        base=self.bias.predict(frame)
        features=[base]
        # Local reconstruction statistics use training answers only.
        for entity in ('user','item'):
            rows=[]
            for key,t in zip(frame[entity],frame.timestamp):
                times,r,e=self.groups[entity].get(key,(np.array([]),np.array([]),np.array([])))
                row=[]
                for days in (7,30,90):
                    lo=np.searchsorted(times,t-days*86400);hi=np.searchsorted(times,t+days*86400,side='right')
                    local=r[lo:hi];res=e[lo:hi]
                    probs=(np.array([(local==0).sum(),(local<=2).sum(),(local==5).sum()])+20*self.priors)/(len(local)+20)
                    row.extend([*probs,res.sum()/(len(res)+20),np.log1p(len(local))])
                rows.append(row)
            features.extend(np.asarray(rows).T)
        graph=self.neighbors._graphs[('jaccard',200,20.)]
        corrections=[]
        for user,item,t in zip(frame.user,frame.item,frame.timestamp):
            ids=self.user_rows.get(user)
            item_id=self.neighbors.items.get(item)
            if ids is None or item_id is None:
                corrections.append([0.,0.,0.]);continue
            item_ids,times,user_residual=self.user_arrays[user]
            weights=graph[item_id,item_ids].toarray().ravel()
            delta=np.abs(times-t)/86400
            decays=np.exp2(-delta[:,None]/np.array([90.,365.,1e20])[None,:])
            w=weights[:,None]*decays
            corrections.append(np.sum(w*user_residual[:,None],axis=0)/(np.abs(w).sum(axis=0)+.1))
        for correction in np.asarray(corrections).T:
            features.append(base+correction)
        return np.column_stack(features)
