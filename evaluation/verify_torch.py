"""New CPU training-runtime verification against NPZ and the same public corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from fabledan.model_torch import load_ckpt
from fabledan.model_np import NumpyModel
from evaluation.reference import observation
from fabledan.encode import encode_decision

def run(checkpoint, npz, cases, output):
    torch.set_num_threads(1)
    model,ck=load_ckpt(checkpoint,device='cpu');model.eval()
    oracle=NumpyModel(npz);state=model.state_dict()
    for name,weight in oracle.w.items():
        np.testing.assert_array_equal(weight,state[name].detach().numpy(),err_msg=name)
    maximum=0.0;exact=0;total=0;decisions=0
    with torch.no_grad():
        for case in json.loads(Path(cases).read_text())['cases']:
            tokens,features=encode_decision(observation(case))
            q,_=model(torch.tensor([tokens]),torch.tensor([len(tokens)]),torch.from_numpy(features[None]))
            expected=q[0].numpy();actual=oracle.q_values(tokens,features)
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4,err_msg=case['id'])
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            exact+=int(np.argmax(actual)==np.argmax(expected));total+=1;decisions+=len(actual)
            if total%100==0:print('torch',total,flush=True)
    report={'schema':'fabledan-torch-reference-v1','torch':torch.__version__,'numpy':np.__version__,
            'device':'cpu','cases':total,'actions':decisions,'npzWeightsExact':True,
            'maxTorchNumpyError':maximum,'exactArgmax':exact,'newTorchRun':True,
            'checkpointMeta':ck['meta'],'checkpointSha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'npzSha256':hashlib.sha256(Path(npz).read_bytes()).hexdigest(),'strengthValidated':False}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--npz',required=True);p.add_argument('--cases',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.checkpoint,a.npz,a.cases,a.out)
