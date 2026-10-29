import argparse,json
from pathlib import Path
import numpy as np
from .audio import load_audio,windows

def torch_scores(clips,model_dir,device='cpu',threads=2):
    import torch
    from transformers import HubertConfig
    from safetensors.torch import load_file
    from .model import HubertRegressor
    torch.set_num_threads(threads)
    model=HubertRegressor(HubertConfig.from_json_file(str(Path(model_dir)/'config.json')))
    model.load_state_dict(load_file(str(Path(model_dir)/'model.safetensors')),strict=True)
    model.to(device).eval();scores=[]
    with torch.inference_mode():
        for clip in clips:
            wave=torch.from_numpy(clip[None]).to(device)
            scores.append(float(model(wave,torch.ones_like(wave,dtype=torch.long))['reg_probs'][0].cpu())*100)
    return scores

def onnx_scores(clips,model_path,threads=2):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs();scores=[]
    for clip in clips:
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
    p.add_argument('audio',type=Path);p.add_argument('--backend',choices=['torch','onnx'],default='torch')
    p.add_argument('--model-dir',type=Path,default=Path('artifacts'))
    p.add_argument('--model',type=Path,help='ONNX file, defaults to MODEL_DIR/model_uint8.onnx')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();audio=load_audio(args.audio);clips=windows(audio)
    if not clips:p.error('At least 3 seconds of audio are required')
    scores=torch_scores(clips,args.model_dir,args.device,args.threads) if args.backend=='torch' else onnx_scores(clips,args.model or args.model_dir/'model_uint8.onnx',args.threads)
    print(json.dumps({'score_percent':float(np.mean(scores)),'window_scores_percent':scores,
        'scored_windows':len(scores),'recording_seconds':len(audio)/16000,'backend':args.backend}))

if __name__=='__main__':main()
