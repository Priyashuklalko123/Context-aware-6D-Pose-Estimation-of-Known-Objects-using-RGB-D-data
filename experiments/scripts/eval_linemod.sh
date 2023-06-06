#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


#python ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed --model_sym ./linemod_models/model_sym.pth --model ./linemod_models/model.pth --refine_model_sym ./linemod_models/refine_sym.pth --refine_model ./linemod_models/refine.pth

#python ./tools/eval_linemod_2.py --dataset_root ./datasets/linemod/Linemod_preprocessed --model_sym ./linemod_models/model_sym.pth --model ./linemod_models/model.pth --refine_model_sym ./linemod_models/refine_sym.pth --refine_model ./linemod_models/refine.pth

python ./tools/eval_linemod_10.py --dataset_root ./datasets/linemod/Linemod_preprocessed --model_sym ./linemod_models/model_sym.pth --model ./linemod_models/model.pth --refine_model_sym ./linemod_models/refine_sym.pth --refine_model ./linemod_models/refine.pth

