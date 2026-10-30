import argparse,json
from pathlib import Path
import numpy as np
from .audio import load_audio,windows

def torch_scores(clips,model_dir,device='cpu',threads=2):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(threads)
    config=json.loads((Path(model_dir)/'config.json').read_text())
    if config.get('architecture')=='ecapa_tdnn':
        from .ecapa import EcapaRegressor
        model=EcapaRegressor(config['width'])
    else:
        from transformers import HubertConfig
        from .model import HubertRegressor
        model=HubertRegressor(HubertConfig.from_dict(config))
    model.load_state_dict(load_file(str(Path(model_dir)/'model.safetensors')),strict=True)
    model.to(device).eval();scores=[]
    with torch.inference_mode():
        for clip in clips:
            if config.get('normalization_per_window'):
                clip=clip/max(float(np.max(np.abs(clip))),1e-8)
            wave=torch.from_numpy(clip[None]).to(device)
            result=model(wave) if config.get('architecture')=='ecapa_tdnn' else model(wave,torch.ones_like(wave,dtype=torch.long))['reg_probs']
            scores.append(float(result[0].cpu())*100)
    return scores

def torchscript_scores(clips,model_path,device='cpu',threads=2):
    import torch
    if torch.device(device).type!='cpu':
        raise ValueError('Released TorchScript runtimes target CPU; use ECAPA Safetensors with --backend torch for GPU inference')
    torch.set_num_threads(threads)
    model=torch.jit.load(str(model_path),map_location=device).eval()
    scores=[]
    with torch.inference_mode():
        for clip in clips:
            score=float(model(torch.from_numpy(clip[None]).to(device)).reshape(-1)[0].cpu())
            if not np.isfinite(score) or not 0<=score<=100:raise RuntimeError('Invalid model output')
            scores.append(score)
    return scores

def onnx_scores(clips,model_path,threads=2,normalize_per_window=False):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs();scores=[]
    for clip in clips:
        if normalize_per_window:clip=clip/max(float(np.max(np.abs(clip))),1e-8)
        feed={}
        for inp in inputs:
            if 'int64' in inp.type:feed[inp.name]=np.ones((1,80000),dtype=np.int64)
            elif 'bool' in inp.type:feed[inp.name]=np.ones((1,80000),dtype=bool)
            else:feed[inp.name]=clip[None].astype(np.float32)
        # This released graph already includes conversion to the 0..100 scale.
        score=float(np.asarray(session.run(None,feed)[0]).reshape(-1)[0])
        if not np.isfinite(score) or not 0<=score<=100:raise RuntimeError('Invalid model output')
        scores.append(score)
    return scores

def main():
    p=argparse.ArgumentParser(description='Research score for perceived voice femininity/passability; not a gender identity detector.')
    p.add_argument('audio',type=Path);p.add_argument('--backend',choices=['torch','onnx','torchscript'],default='torch')
    p.add_argument('--model-dir',type=Path,default=Path('artifacts'))
    p.add_argument('--model',type=Path,help='ONNX or TorchScript file; defaults to MODEL_DIR/model_uint8.onnx or model.ts')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();audio=load_audio(args.audio);clips=windows(audio)
    if not clips:p.error('At least 3 seconds of audio are required')
    if args.backend=='torch':scores=torch_scores(clips,args.model_dir,args.device,args.threads)
    elif args.backend=='torchscript':scores=torchscript_scores(clips,args.model or args.model_dir/'model.ts',args.device,args.threads)
    else:
        config_path=args.model_dir/'config.json'
        normalize=json.loads(config_path.read_text()).get('normalization_per_window',False) if config_path.exists() else False
        scores=onnx_scores(clips,args.model or args.model_dir/'model_uint8.onnx',args.threads,normalize)
    print(json.dumps({'score_percent':float(np.mean(scores)),'window_scores_percent':scores,
        'scored_windows':len(scores),'recording_seconds':len(audio)/16000,'backend':args.backend}))

if __name__=='__main__':main()
