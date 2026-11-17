"""Predict next-event gaps using causal history and the known destination book."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DAY=86400.

def prepare(data_dir):
    raw=pd.read_csv(Path(data_dir)/'task_6_train.csv',dtype={'user':'string','item':'string'}).sort_values(['user','timestamp'],kind='stable').reset_index(drop=True)
    g=raw.groupby('user',sort=False)
    raw['gap']=g.timestamp.diff();raw['target']=g.timestamp.shift(-1)-raw.timestamp
    raw['next_item']=g.item.shift(-1)
    x=pd.DataFrame(index=raw.index);g=raw.groupby('user',sort=False).gap
    print('Building causal Task 6 histories...',flush=True)
    for lag in (0,1,2,3):x[f'lag_{lag}']=raw.gap if lag==0 else g.shift(lag)
    for window in (3,10,30):
        for stat in ('mean','median','std'):
            x[f'{stat}_{window}']=g.transform(lambda s:getattr(s.rolling(window,min_periods=1),stat)())
    x['history_mean']=g.transform(lambda s:s.expanding().mean())
    x['history_median']=g.transform(lambda s:s.expanding().median())
    x['count']=raw.groupby('user',sort=False).cumcount()
    x['elapsed']=raw.timestamp-raw.groupby('user').timestamp.transform('min')
    x['rate']=x['count']/(x.elapsed/DAY+1)
    x['recent_ratio']=x.mean_3/(x.history_mean+1)
    x['timestamp']=raw.timestamp
    for period,name in ((DAY,'day'),(7*DAY,'week'),(365.25*DAY,'year')):
        phase=2*np.pi*raw.timestamp/period
        x[name+'_sin']=np.sin(phase);x[name+'_cos']=np.cos(phase)
    for name in ['lag_0','lag_1','history_mean','history_median','mean_3','median_3','mean_10','elapsed']:
        x['log_'+name]=np.log1p(x[name].clip(lower=0))
    valid=raw.target.notna()
    hold=raw[valid].groupby('user',sort=False).tail(1).index.to_numpy()
    train=raw.index[valid].difference(hold).to_numpy()
    tune,evaluation=np.array_split(np.random.default_rng(42).permutation(hold),2)
    print('Training/tuning/evaluation rows:',len(train),len(tune),len(evaluation),flush=True)
    return dict(raw=raw,x=x,valid=valid,train=train,tune=tune,evaluation=evaluation,data_dir=Path(data_dir))

def features(state,reference,queries,items=None,loo=False):
    raw=state['raw'];gap=raw.loc[reference,'target'].clip(upper=730*DAY)
    arrival=raw.loc[reference,'timestamp']+raw.loc[reference,'target']
    # Robust date statistics avoid a few anomalous ancient/future timestamps.
    date_low,date_high=arrival.quantile([.001,.999]).to_numpy()
    arrival=arrival.clip(lower=date_low,upper=date_high)
    date_center=float(arrival.mean())
    date_offset=(arrival-date_center)/DAY
    table=pd.DataFrame({'item':raw.loc[reference,'next_item'].to_numpy(),'gap':gap.to_numpy(),'log_gap':np.log1p(gap.to_numpy()),'date':date_offset.to_numpy()})
    stats=table.groupby('item').agg(total=('gap','sum'),log_total=('log_gap','sum'),count=('gap','size'),date_total=('date','sum'))
    keys=raw.loc[queries,'next_item'] if items is None else pd.Series(np.asarray(items),index=queries)
    count=keys.map(stats['count']).fillna(0).to_numpy(float)
    total=keys.map(stats.total).fillna(0).to_numpy(float)
    logs=keys.map(stats.log_total).fillna(0).to_numpy(float)
    dates=keys.map(stats.date_total).fillna(0).to_numpy(float)
    if loo:
        own=raw.loc[queries,'target'].clip(upper=730*DAY).to_numpy(float)
        count-=1;total-=own;logs-=np.log1p(own)
        own_arrival=(raw.loc[queries,'timestamp']+raw.loc[queries,'target']).clip(lower=date_low,upper=date_high)
        dates-=(own_arrival.to_numpy()-date_center)/DAY
    x=state['x'].loc[queries].copy()
    x['next_book_gap']=(total+30*gap.mean())/(count+30)
    x['next_book_log_gap']=(logs+30*np.log1p(gap).mean())/(count+30)
    x['next_book_count']=np.log1p(np.maximum(count,0))
    mean_arrival=date_center+DAY*dates/(count+5)
    offset=(mean_arrival-raw.loc[queries,'timestamp'].to_numpy())/DAY
    x['next_book_date_offset']=offset
    x['next_book_date_gap']=np.maximum(offset,0)*DAY
    return x

def fit(state,reference,config):
    # Later sequence positions resemble the last-event prediction task.
    indices=state['raw'].loc[reference].groupby('user',sort=False).tail(30).index.to_numpy()
    x=features(state,reference,indices,loo=True)
    y=state['raw'].loc[indices,'target'].to_numpy(float)
    leaves,cap=config[:2]
    if len(config)>2 and not config[2]:
        x=x.drop(columns=['next_book_date_offset','next_book_date_gap'])
    if cap is not None:y=np.minimum(y,cap*DAY)
    model=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=leaves,min_samples_leaf=100,l2_regularization=30,learning_rate=.05,early_stopping=False,random_state=42)
    model.fit(x,y)
    return model

def evaluate(state):
    raw=state['raw'];reference=state['train'];t=state['tune'];e=state['evaluation']
    xt=features(state,reference,t);xe=features(state,reference,e)
    yt=raw.loc[t,'target'].to_numpy(float);ye=raw.loc[e,'target'].to_numpy(float)
    configs={f'boost_{leaves}_cap_{cap}':(leaves,cap) for leaves in (15,31) for cap in (365,730,None)}
    configs.update({'reference_no_date_15':(15,None,False),'reference_no_date_31':(31,730,False)})
    pt={};pe={}
    for name,config in configs.items():
        print('Fitting Task 6',name,flush=True)
        model=fit(state,reference,config)
        pt[name]=np.maximum(0,model.predict(xt[model.feature_names_in_]));pe[name]=np.maximum(0,model.predict(xe[model.feature_names_in_]))
        print('Tuning RMSE:',mean_squared_error(yt,pt[name])**.5,flush=True)
    fallback=raw.loc[reference,'target'].median()
    baselines={'baseline_user_median':'history_median','user_mean':'history_mean','recent_median':'median_3'}
    for name,column in baselines.items():
        pt[name]=state['x'].loc[t,column].fillna(fallback).to_numpy()
        pe[name]=state['x'].loc[e,column].fillna(fallback).to_numpy()
    best=min(pt,key=lambda name:mean_squared_error(yt,pt[name]))
    chosen=(best,best,1.);best_score=mean_squared_error(yt,pt[best])
    top=sorted(pt,key=lambda name:mean_squared_error(yt,pt[name]))[:3]
    for a in top:
        for b in top:
            for w in np.linspace(0,1,11):
                score=mean_squared_error(yt,w*pt[a]+(1-w)*pt[b])
                if score<best_score:best_score=score;chosen=(a,b,float(w))
    a,b,w=chosen
    pe['selected_blend']=w*pe[a]+(1-w)*pe[b]
    pt['selected_blend']=w*pt[a]+(1-w)*pt[b]
    baseline=mean_squared_error(ye,pe['baseline_user_median'])
    results=pd.DataFrame([dict(model=name,tune_mse=mean_squared_error(yt,pt[name]),evaluation_mse=mean_squared_error(ye,pred),evaluation_rmse=mean_squared_error(ye,pred)**.5) for name,pred in pe.items()])
    results['mse_reduction_%']=100*(1-results.evaluation_mse/baseline)
    print(results.sort_values('evaluation_mse').to_string(index=False),flush=True)
    print('Selected on tuning users:',chosen,flush=True)
    print('MSE is seconds squared; RMSE is seconds. These are different metrics.',flush=True)
    errors=pd.DataFrame({'target':ye,'prediction':pe['selected_blend']})
    errors['gap_days']=pd.cut(errors.target/DAY,[-1,1,7,30,90,365,np.inf])
    errors['squared_error']=(errors.target-errors.prediction)**2
    report=errors.groupby('gap_days',observed=True).agg(rows=('target','size'),mse=('squared_error','mean'),total_error=('squared_error','sum'))
    report['error_share_%']=100*report.total_error/errors.squared_error.sum()
    print('Evaluation error by true gap in days:',report.to_string(),sep='\n',flush=True)
    state.update(configs=configs,baselines=baselines,chosen=chosen,results=results)
    return results

def predict(state,output_dir):
    raw=state['raw'];reference=raw.index[state['valid']].to_numpy()
    test=pd.read_csv(state['data_dir']/'task_6_test.csv',dtype={'user':'string','item':'string'})
    lookup=raw.reset_index().groupby('user',sort=False).tail(1).set_index('user')['index']
    if test.user.map(lookup).isna().any():raise ValueError('Unexpected unseen test user')
    indices=test.user.map(lookup).to_numpy(int)
    x=features(state,reference,indices,items=test.item)
    predictions={}
    for name in set(state['chosen'][:2]):
        if name in state['configs']:
            print('Full-data Task 6 fit:',name,flush=True)
            model=fit(state,reference,state['configs'][name]);predictions[name]=np.maximum(0,model.predict(x[model.feature_names_in_]))
        else:
            predictions[name]=state['x'].loc[indices,state['baselines'][name]].fillna(raw.loc[reference,'target'].median()).to_numpy()
    a,b,w=state['chosen'];test['delta']=w*predictions[a]+(1-w)*predictions[b]
    if not np.isfinite(test.delta).all():raise ValueError('Non-finite predictions')
    output_dir=Path(output_dir);output_dir.mkdir(exist_ok=True)
    test[['user','item','delta']].to_csv(output_dir/'task_6_predictions.csv',index=False)
    state['results'].to_csv(output_dir/'task6_validation.csv',index=False)
    print('Wrote',output_dir/'task_6_predictions.csv',flush=True)
    return test

if __name__=='__main__':
    directory=Path(__file__).resolve().parent
    state=prepare(directory)
    evaluate(state)
