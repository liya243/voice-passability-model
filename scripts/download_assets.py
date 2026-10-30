"""Download public release assets with SHA256 verification; no audio uploads."""
import argparse,hashlib,json,os,tempfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(1024*1024):digest.update(block)
    return digest.hexdigest()

def download(name,output,manifest,target_name=None):
    output.mkdir(parents=True,exist_ok=True);target=output/(target_name or name)
    wanted=manifest['assets'][name]['sha256']
    if target.exists() and sha256(target)==wanted:return target
    url=f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/{name}'
    request=urllib.request.Request(url,headers={'User-Agent':'voice-passability-assets/0.1'})
    temporary=target.with_name(target.name+'.download')
    digest=hashlib.sha256()
    try:
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as handle:
            while block:=response.read(1024*1024):digest.update(block);handle.write(block)
        if digest.hexdigest()!=wanted:raise RuntimeError('SHA256 mismatch for '+name)
        os.replace(temporary,target)
    finally:
        if temporary.exists():temporary.unlink()
    print(target)
    return target

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['torch','onnx','torchscript'])
    p.add_argument('--variant',help='Named v0.2.0 model; see model_zoo.json. Omit for the original v0.1.0 model.')
    p.add_argument('--dataset',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'artifacts')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');args=p.parse_args()
    if not args.backend and not args.dataset:p.error('Choose --backend or --dataset')
    manifest=json.loads((ROOT/'assets.json').read_text())
    if args.variant:
        zoo=json.loads((ROOT/'model_zoo.json').read_text())
        model=zoo['models'].get(args.variant)
        if not model:p.error('Unknown variant; choose from '+', '.join(zoo['models']))
        if args.backend not in model['backends']:p.error('Choose a supported backend: '+', '.join(model['backends']))
        model_manifest=json.loads((ROOT/'assets-v0.2.0.json').read_text())
        for entry in model['backends'][args.backend]:
            download(entry['asset'],args.output,model_manifest,entry['target'])
    else:
        if args.backend=='torchscript':p.error('TorchScript requires --variant expert-mini or original-mini')
        for name in (['model.safetensors','config.json'] if args.backend=='torch' else ['model_uint8.onnx'] if args.backend=='onnx' else []):
            download(name,args.output,manifest)
    if args.dataset:
        archive=download('features-and-ratings-v0.1.0.zip',args.output,manifest)
        allowed={'ecapa_features.npz','records.csv.gz','individual_ratings.csv.gz','cluster_ratings.csv.gz',
                 'adopted_cluster_ratings.csv.gz','dataset_manifest.json','DATASET_CARD.md','LICENSE_DATA.md'}
        args.data_dir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=allowed:raise RuntimeError('Unexpected dataset archive contents')
            for name in z.namelist():
                with z.open(name) as source,(args.data_dir/name).open('wb') as target:
                    while block:=source.read(1024*1024):target.write(block)
        print(args.data_dir)

if __name__=='__main__':main()
