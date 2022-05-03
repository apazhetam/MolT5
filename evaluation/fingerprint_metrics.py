'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')

parser.add_argument('--morgan_r', type=int, default=2, help='morgan fingerprint radius')

args = parser.parse_args()


outputs = []
bad_mols = 0

with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        try:
            gt_smi = line['ground truth']
            ot_smi = line['output']
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append((line['description'], gt_m, ot_m))
        except:
            bad_mols += 1
print('validity:', len(outputs)/(len(outputs)+bad_mols))


MACCS_sims = []
morgan_sims = []
RDK_sims = []

enum_list = outputs

for i, (desc, gt_m, ot_m) in enumerate(enum_list):
    
    if i % 100 == 0: print(i, 'processed.')



    MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
    RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
    morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,args.morgan_r), AllChem.GetMorganFingerprint(ot_m, args.morgan_r)))



print('Average MACCS Similarity:', np.mean(MACCS_sims))
print('Average RDK Similarity:', np.mean(RDK_sims))
print('Average Morgan Similarity:', np.mean(morgan_sims))