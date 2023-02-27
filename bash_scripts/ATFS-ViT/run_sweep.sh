#!/bin/bash

algorithms=ATFSViT
alpha=0.1
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
backbone=DeitSmall # DeitSmall, T2T14
data_dir=/path/to/data/
output_dir=/where/to/save/${dataset}/${algorithm}/${backbone}


for n_layers in 1 2 3 4  # number of random layers to apply TFS (n in the paper)
do
    for d_rate in 0.1 0.3 0.5 0.8 # the rate of token selection and replacement (d in the paper)
        do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
            --output_dir=${output_dir}/sweep_drate_${d_rate}_nlay_${n_layers}  --command_launcher multi_gpu --algorithms ${algorithms}  \
            --single_test_envs  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
            --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"num_layers\":$n_layers,\"d_rate\":$d_rate,\"alpha\":$alpha}"""
        done
    done
done