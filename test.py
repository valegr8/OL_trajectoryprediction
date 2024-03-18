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

from datasets import ArgoverseV2Dataset
from predictors import QCNet

from transforms import TargetBuilder

import os
import pandas as pd

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='test',transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))

    metrics_savepath='test_metrics.csv'

    # Check if the file exists
    if os.path.exists(metrics_savepath):
        # If it exists, delete it
        os.remove(metrics_savepath)


    # Manually create a header dataframe with the column names
    header_df = pd.DataFrame({'scenario_id': ['scenario_id'], 'val_Brier': ['val_Brier'], 'val_minADE': ['val_minADE'], 'val_minAHE': ['val_minAHE'], 'val_minFDE': ['val_minFDE'], 'val_minFHE': ['val_minFHE'], 'val_minMR': ['val_minMR']})


    # Append the header dataframe to the main dataframe
    header_df.to_csv('test_metrics.csv', index=False, header=False)

    test_dataset = test_dataset[:1] 
    print('------------------------------------------------------------------------------------------------------------------')
    print('test dataset:', test_dataset)
    print("test dataset: ", len(test_dataset))
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    
    el=next(iter(dataloader))
    print(el)

    print(el['agent']['position'] )
    print('------------------------------------------------------------------------------------------------------------------')

    print(el['agent']['target'] )

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp', enable_checkpointing=False)
    trainer.test(model, dataloader)
