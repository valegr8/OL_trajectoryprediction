# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import os

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import torch

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    # Create a TensorBoardLogger instance
    logger = TensorBoardLogger("lightning_logs", name="my_experiment")

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    print('model his steps:', model.num_historical_steps)

    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='olval',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))


    print('LEN DATASET: ', len(val_dataset))

    print('MODEL PARAMS: ', sum(p.numel() for p in model.parameters()))

    # model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    metrics_savepath='val_metrics.csv'

    # Check if the file exists
    if os.path.exists(metrics_savepath):
        # If it exists, delete it
        os.remove(metrics_savepath)

    # Manually create a header dataframe with the column names
    header_df = pd.DataFrame({'scenario_id': ['scenario_id'], 'val_Brier': ['val_Brier'], 'val_minADE': ['val_minADE'], 'val_minAHE': ['val_minAHE'], 'val_minFDE': ['val_minFDE'], 'val_minFHE': ['val_minFHE'], 'val_minMR': ['val_minMR']})


    # Append the header dataframe to the main dataframe
    header_df.to_csv('val_metrics.csv', index=False, header=False)

    # val_dataset = val_dataset[:1] 
    # print('------------------------------------------------------------------------------------------------------------------')
    # print('val dataset:', val_dataset)

    # Clear the memory used by the GPU
    torch.cuda.empty_cache()


    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    # print(next(iter(dataloader)))
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp', logger=logger)
    trainer.validate(model, dataloader)
