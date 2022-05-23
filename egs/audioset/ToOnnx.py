# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import os
import sys
import argparse
import torch
import torch.onnx
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel


# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'
if __name__ == '__main__':
    checkpoint_path = '/home/mo/workspace/mo/ast/pretrained_models/audioset_10_10_0.4593.pth'
    input_tdim = 1024
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    BATCH_SIZE = 1
    feats_data = torch.randn(BATCH_SIZE, input_tdim, 128, dtype=torch.float32).to(torch.device("cuda:0"))
    audio_model.eval()                                      # set the eval model
    torch.onnx.export(audio_model.module, feats_data, "audioset_10_10_0.4593_onnx_model.onnx", verbose=True)
    print('Done !')
