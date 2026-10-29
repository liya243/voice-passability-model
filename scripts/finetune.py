"""Minimal local-waveform adaptation example; not the historical training recipe."""
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from transformers import HubertConfig
from voice_passability.audio import load_audio,windows
from voice_passability.model import HubertRegressor

def main():
    p=argparse.ArgumentParser(description='Fine-tune on your own permitted audio, using a CSV with audio_path,score_percent.')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model-dir',type=Path,default=Path('artifacts'))
    p.add_argument('--output',type=Path,default=Path('adapted_model'));p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--unfreeze-last',type=int,default=0)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    config=HubertConfig.from_json_file(str(args.model_dir/'config.json'))
    model=HubertRegressor(config);model.load_state_dict(load_file(str(args.model_dir/'model.safetensors')),strict=True)
    if args.epochs<1 or not 0<=args.unfreeze_last<=len(model.backbone.encoder.layers):
        p.error('epochs must be positive; unfreeze-last must fit the encoder layer count')
    for parameter in model.parameters():parameter.requires_grad=False
    for parameter in model.heads.reg_head.parameters():parameter.requires_grad=True
    if args.unfreeze_last:
        for layer in model.backbone.encoder.layers[-args.unfreeze_last:]:
            for parameter in layer.parameters():parameter.requires_grad=True
    model.to(args.device).train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5)
    with args.manifest.open(newline='') as handle:rows=list(csv.DictReader(handle))
    if not rows:p.error('Manifest is empty')
    for epoch in range(args.epochs):
        random.shuffle(rows);losses=[]
        for row in rows:
            score=float(row['score_percent'])
            if not 0<=score<=100:raise ValueError('score_percent must be in [0,100]')
            path=Path(row['audio_path'])
            if not path.is_absolute():path=args.manifest.parent/path
            chunks=windows(load_audio(path))
            if not chunks:raise ValueError('A recording has less than 3 seconds of eligible audio')
            wave=torch.from_numpy(random.choice(chunks)[None]).to(args.device)
            prediction=model(wave,torch.ones_like(wave,dtype=torch.long))['reg_probs']
            target=torch.tensor([score/100],device=args.device)
            loss=torch.nn.functional.smooth_l1_loss(prediction,target,beta=.05)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({'epoch':epoch+1,'mean_training_loss':float(np.mean(losses))}))
    args.output.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(args.output/'model.safetensors'))
    config.to_json_file(str(args.output/'config.json'))
    print('Saved adapted weights. Evaluate on a separate grouped validation set before interpreting improvements.')

if __name__=='__main__':main()
