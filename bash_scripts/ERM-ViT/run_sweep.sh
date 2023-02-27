#!/bin/bash

algorithms=ERM_ViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
backbone=DeitSmall # DeitSmall, T2T14
data_dir=/path/to/data/
output_dir=/where/to/save/${dataset}/${algorithm}/${backbone}

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_test_envs  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done
