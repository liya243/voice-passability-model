"""A reproducible CPU baseline on public ECAPA features and individual ratings."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np

def main():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    with np.load(args.dataset/'ecapa_features.npz',allow_pickle=False) as z:
        ids=z['record_ids'].astype(str);features=z['embeddings']
    with gzip.open(args.dataset/'records.csv.gz','rt',encoding='utf-8') as f:rows={r['record_id']:r for r in csv.DictReader(f)}
    available=np.array([rows[r]['individual_mean_percent']!='' for r in ids])
    x=features[available];selected=ids[available]
    y=np.array([float(rows[r]['individual_mean_percent']) for r in selected])
    split=np.array([rows[r]['split'] for r in selected])
    train,val,test=[split==part for part in ['train','validation','test']]
    candidates=[]
    for alpha in [.1,1,10,100]:
        model=Ridge(alpha=alpha).fit(x[train],y[train])
        error=mean_absolute_error(y[val],np.clip(model.predict(x[val]),0,100))
        candidates.append((float(error),alpha,model))
    _,alpha,model=min(candidates,key=lambda r:r[:2])
    result={'target':'mean genuine individual rating per recording (equal rater weight)',
            'selected_alpha':alpha,'selection':'validation only','label_source':'individual; no weak or adopted labels',
            'split_counts':{part:int((split==part).sum()) for part in ['train','validation','test']},
            'mae_points':{part:float(mean_absolute_error(y[split==part],np.clip(model.predict(x[split==part]),0,100)))
                          for part in ['train','validation','test']}}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
