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
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.core.hooks import ModelHooks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder

import matplotlib.pyplot as plt
import pandas as pd

import torch.optim as optim

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()

        # print('learning rate: ', self.lr)
        # params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        # self.optimizer = torch.optim.AdamW(params, lr=0.005, weight_decay=self.weight_decay)
        # print('OPTIMIZER INItiALizED')

        self.optimizer_MLP()

        self.save_loss_cls = []
        self.save_loss_refine = []
        self.save_loss_propose = []

    def optimizer_MLP(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name

                
                if ('decoder.to_loc_propose_pos.mlp' in full_param_name) or ('decoder.to_scale_propose_pos.mlp' in full_param_name) or ('decoder.to_pi.mlp' in full_param_name):
                    print(full_param_name)
                    if 'bias' in param_name:
                        no_decay.add(full_param_name)
                    elif 'weight' in param_name:
                        if isinstance(module, whitelist_weight_modules):
                            decay.add(full_param_name)
                        elif isinstance(module, blacklist_weight_modules):
                            no_decay.add(full_param_name)
                    elif not ('weight' in param_name or 'bias' in param_name):
                        no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        self.optimizer = optimizer

        print('OPTIMIZER CHANGED')

        return optimizer


    def forward(self, data: HeteroData):
       # print(data)
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred
    
    def set_num_historical_steps(self, num_historical_steps):
        self.num_historical_steps = num_historical_steps
        self.encoder.agent_encoder.set_num_historical_steps(num_historical_steps)
        self.encoder.map_encoder.set_num_historical_steps(num_historical_steps)
        self.decoder.set_num_historical_steps(num_historical_steps)



    # def training_step(self,
    #                   data,
    #                   batch_idx,
    #                   online_learning,
    #                   ol_time_slice: int):
    #     # print('TRAINING STEP: ', self.num_historical_steps)
    #     torch.cuda.empty_cache()
    #     if isinstance(data, Batch):
    #         data['agent']['av_index'] += data['agent']['ptr'][:-1]

    #     # the mask is needed only to calculate the loss of ceirtain elements, for stabilization, so that with the background thep we don update all weights
    #     reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
    #     cls_mask = data['agent']['predict_mask'][:, -1]

    #     pred = self(data)
    #     if self.output_head:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['loc_propose_head'],
    #                                   pred['scale_propose_pos'][..., :self.output_dim],
    #                                   pred['conc_propose_head']], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['loc_refine_head'],
    #                                  pred['scale_refine_pos'][..., :self.output_dim],
    #                                  pred['conc_refine_head']], dim=-1)
    #     else:
    #         traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
    #                                   pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
    #         traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
    #                                  pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
    #     pi = pred['pi']

    #     if online_learning:
    #         next_gt = self.num_historical_steps+ol_time_slice
    #         gt = torch.cat((data['agent']['position'][:, self.num_historical_steps:next_gt].contiguous(),), dim=1)
    #         # print('gt pos', gt.shape)

    #         # gt_head_a = torch.cat((data['agent']['heading'][:, self.num_historical_steps:next_gt].contiguous(),), dim=1)
    #         # print('gt pos', gt_head_a.shape)

    #         # print('traj propose before shape', traj_propose.shape)
    #         traj_propose = traj_propose[...,:ol_time_slice, :] 
    #         # print('traj propose after shape', traj_propose.shape)
    #         # print('traj refine shape', traj_refine.shape)
    #         traj_refine = traj_refine[...,:ol_time_slice, :] 
    #         # print('traj refine after', traj_refine.shape)
    #         reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:next_gt]
    #         # print('reg mask shape: ', reg_mask.shape)
    #         # print('cls_mask_shape:', cls_mask.shape)

    #     else:
    #         gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

    #     # print('gt shape: ', gt[..., :self.output_dim].shape)

    #     if online_learning:
    #         l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
    #                             gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1)).sum(dim=-1)
            
    #         best_mode = l2_norm.argmin(dim=-1)
    #         traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
    #         traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
    #         reg_loss_propose = self.reg_loss(traj_propose_best,
    #                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1)
    #         reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #         reg_loss_propose = reg_loss_propose.mean()
    #         reg_loss_refine = self.reg_loss(traj_refine_best,
    #                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1)
    #         reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #         reg_loss_refine = reg_loss_refine.mean()
    #         cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
    #                                 target=gt[:, -1:, :self.output_dim + self.output_head],
    #                                 prob=pi,
    #                                 mask=reg_mask[:, -1:])
    #     else:
    #         # print(gt[..., :self.output_dim].unsqueeze(1).shape)
    #         l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
    #                             gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
    #         best_mode = l2_norm.argmin(dim=-1)
    #         traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
    #         traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
    #         reg_loss_propose = self.reg_loss(traj_propose_best,
    #                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #         reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #         reg_loss_propose = reg_loss_propose.mean()
    #         reg_loss_refine = self.reg_loss(traj_refine_best,
    #                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
    #         reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
    #         reg_loss_refine = reg_loss_refine.mean()
    #         cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
    #                                 target=gt[:, -1:, :self.output_dim + self.output_head],
    #                                 prob=pi,
    #                                 mask=reg_mask[:, -1:]) * cls_mask
    #     cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
    #     self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
    #     self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
    #     self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

    #     # Set requires_grad to True
    #     reg_loss_propose.requires_grad_()
    #     reg_loss_refine.requires_grad_()
    #     cls_loss.requires_grad_()


    #     loss = reg_loss_propose + reg_loss_refine + cls_loss

    #     loss.requires_grad_()
    #     return loss
        
    def training_step(self,
                      data,
                      batch_idx,
                      online_learning,
                      ol_time_slice: int,
                      final_his_step):
        # print('TRAINING STEP: ', self.num_historical_steps)
        torch.cuda.empty_cache()
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        # the mask is needed only to calculate the loss of ceirtain elements, for stabilization, so that with the background thep we don update all weights
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        torch.set_grad_enabled(True)
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        if online_learning:
            ol_time_slice = final_his_step-self.num_historical_steps
            gt = torch.cat((data['agent']['position'][:, self.num_historical_steps:final_his_step].contiguous(),), dim=1)
            # print('gt pos', gt.shape)

            # gt_head_a = torch.cat((data['agent']['heading'][:, self.num_historical_steps:next_gt].contiguous(),), dim=1)
            # print('gt pos', gt_head_a.shape)

            # print('traj propose before shape', traj_propose.shape)
            traj_propose = traj_propose[...,:ol_time_slice, :] 
            # print('traj propose after shape', traj_propose.shape)
            # print('traj refine shape', traj_refine.shape)
            traj_refine = traj_refine[...,:ol_time_slice, :] 
            # print('traj refine after', traj_refine.shape)
            reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:final_his_step]
            # print('reg mask shape: ', reg_mask.shape)
            # print('cls_mask_shape:', cls_mask.shape)

        else:
            gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        # print('gt shape: ', gt[..., :self.output_dim].shape)

        if online_learning:
            l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                                gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1)).sum(dim=-1)
            
            best_mode = l2_norm.argmin(dim=-1)
            traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
            traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
            reg_loss_propose = self.reg_loss(traj_propose_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1)
            reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_propose = reg_loss_propose.mean()
            reg_loss_refine = self.reg_loss(traj_refine_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1)
            reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_refine = reg_loss_refine.mean()

            all_ones_mask = torch.ones_like(reg_mask)
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                    target=gt[:, -1:, :self.output_dim + self.output_head],
                                    prob=pi,
                                    mask=all_ones_mask[:, -1])
        else:
            # print(gt[..., :self.output_dim].unsqueeze(1).shape)
            l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                                gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
            best_mode = l2_norm.argmin(dim=-1)
            traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
            traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
            reg_loss_propose = self.reg_loss(traj_propose_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_propose = reg_loss_propose.mean()
            reg_loss_refine = self.reg_loss(traj_refine_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_refine = reg_loss_refine.mean()
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                    target=gt[:, -1:, :self.output_dim + self.output_head],
                                    prob=pi,
                                    mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        # Set requires_grad to True
        # reg_loss_propose.requires_grad_()
        # reg_loss_refine.requires_grad_()
        # cls_loss.requires_grad_()

        self.save_loss_cls.append(cls_loss.cpu())
        self.save_loss_refine.append(reg_loss_refine.cpu())
        self.save_loss_propose.append(reg_loss_propose.cpu())

        loss = reg_loss_propose + reg_loss_refine + cls_loss


        # print('[TRAING STEP] LOSS:', loss)

        loss.requires_grad_()
        return loss
    
    def online_learning_update(self, data, initial_hstorical_steps, final_his_step, batch_idx, ol_time_slice):
         # Clear the memory used by the GPU
        torch.cuda.empty_cache()

        # enable grads
        torch.set_grad_enabled(True)
        for i in range(initial_hstorical_steps, final_his_step, ol_time_slice):
            self.set_num_historical_steps(i)
            # print(i, '-',initial_hstorical_steps, '-', final_his_step,'-------------------------------------------------------------------')
            loss = self.training_step(data, batch_idx, online_learning = True, ol_time_slice=ol_time_slice, final_his_step = final_his_step)
            # print('[online_learning STEP] LOSS:', loss)
            # print('loss:', loss)

            # clear gradients
            self.optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # print('self.optimizer', self.optimizer)

            # Update model parameters
            self.optimizer.step()

            # Zero the gradients
            self.optimizer.zero_grad()
        self.set_num_historical_steps(final_his_step)


    # def online_learning_update(self, data, initial_hstorical_steps, final_his_step, batch_idx, ol_time_slice):
    #      # Clear the memory used by the GPU
    #     torch.cuda.empty_cache()

    #     # enable grads
    #     torch.set_grad_enabled(True)
    #     for i in range(initial_hstorical_steps, final_his_step,50):
    #         self.set_num_historical_steps(i)
    #         # print(i, '-',initial_hstorical_steps, '-', final_his_step,'-------------------------------------------------------------------')
    #         loss = self.training_step(data, batch_idx, online_learning = True, ol_time_slice=ol_time_slice)

    #         # print('loss:', loss)

    #         # clear gradients
    #         self.optimizer.zero_grad()

    #         # Compute gradients
    #         loss.backward()

    #         # Update model parameters
    #         self.optimizer.step()

    #         # Zero the gradients
    #         self.optimizer.zero_grad()
    #     self.set_num_historical_steps(final_his_step)
        

    def validation_step(self,
                        data,
                        batch_idx):
        # Clear the memory used by the GPU
        torch.cuda.empty_cache()

        

        # Access the weights of the model's layers
        for module_name, module in self.named_modules():
            if 'decoder' in module_name:
                # print(module_name)
                module.train()
            # for param_name, param in module.named_parameters():
            #         full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name

                    # if 'decoder.to_pi.mlp.0.weight' in full_param_name:
                        # print('--------------------------------------------------------------------------------------------------------------')
                        # print(f"Weights of {full_param_name}:")
                        # print(param.data)


        initial_hstorical_steps = 20
        final_his_step = 50

        self.online_learning_update(data, initial_hstorical_steps, final_his_step, batch_idx, ol_time_slice = 5)


        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        #print(data['scenario_id'])

        # Create an empty DataFrame
        df_metrics = pd.DataFrame(columns=['scenario_id','val_Brier','val_minADE','val_minAHE', 'val_minFDE','val_minFHE','val_minMR'])

        # Assign the tensor to the 'scenario_id' column
        df_metrics['scenario_id'] = data['scenario_id']

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        # print('heading: last element od data(agent)(target): ',data['agent']['target'][..., -1:].shape)

        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        # print(traj_propose[..., :self.output_dim].shape)
        # print(gt[..., :self.output_dim].unsqueeze(1).shape)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)


        self.save_loss_cls.append(cls_loss.cpu())
        self.save_loss_refine.append(reg_loss_refine.cpu())
        self.save_loss_propose.append(reg_loss_propose.cpu())

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        brier = self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_ade = self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_ahe = self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_fde = self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,valid_mask=valid_mask_eval, df_metrics=df_metrics)

        min_fhe = self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_mr = self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        

        # Clear the memory used by the GPU
        torch.cuda.empty_cache()

        # df_metrics['val_Brier'] = brier
        # df_metrics['val_minADE'] = min_ade
        # df_metrics['val_minAHE'] = min_ahe
        # df_metrics['val_minFDE'] = min_fde
        # df_metrics['val_minFHE'] = min_fhe
        # df_metrics['val_minMR'] = min_mr

        # print(df_metrics)

        # save to csv
        df_metrics.to_csv('val_metrics.csv', mode='a', index=False, header=False)


        # print('val_Brier: ', brier)
        # print('val_minADE: ', min_ade)
        # print('val_minAHE: ', min_ahe)
        # print('val_minFDE: ', min_fde)
        # print('val_minFHE: ', min_fhe)
        # print('val_minMR: ', min_mr)

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        # print('NOT TO SUBMIT: ',traj_eval)
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        # print('-----------------------------------------------------------------------------------------')
        # print('TO SUBMIT: ',traj_eval)
        # print('-----------------------------------------------------------------------------------------')
        # print('mask: ', eval_mask)
        # print('probabilities: ', pi_eval)
        traj_eval= traj_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = {eval_id[i]: (traj_eval[i], pi_eval[i].cpu().numpy())}
            else:
                self.test_predictions[data['scenario_id'][i]] = {eval_id[0]: (traj_eval[0], pi_eval[0].cpu().numpy())}
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))


        
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def on_validation_end(self):
        if self.dataset == 'argoverse_v2':
            save_path = Path(self.submission_dir) / f'{self.submission_file_name}_val.parquet'
            ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            print('saved in: ', save_path)

            

            # Assuming save_loss_cls, save_loss_refine, and save_loss_propose are lists containing loss values

            # Generate x-axis values (iterations or epochs)
            iterations = range(len(self.save_loss_cls))

            # Plotting the losses
            plt.plot(iterations, self.save_loss_cls, label='Classification Loss')
            plt.plot(iterations, self.save_loss_refine, label='Refinement Loss')
            plt.plot(iterations, self.save_loss_propose, label='Proposal Loss')

            # Adding labels and title
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Losses over iterations')
            plt.legend()  # Show legend
            plt.grid(True)  # Show grid

            # Show plot
            plt.show()

        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)

        # print("DATA",data)
        # print("PRED",pred)

        # Create an empty DataFrame
        df_metrics = pd.DataFrame(columns=['scenario_id','val_Brier','val_minADE','val_minAHE', 'val_minFDE','val_minFHE','val_minMR'])

        # Assign the tensor to the 'scenario_id' column
        df_metrics['scenario_id'] = data['scenario_id']

        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        # print(len(data['agent']['target'][..., :self.output_dim]))
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        # print(len(gt))
        # print(self.num_historical_steps)
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]


        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        print(eval_mask)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        valid_mask_eval = reg_mask[eval_mask]
        gt_eval = gt[eval_mask]

        brier = self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_ade = self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_ahe = self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_fde = self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,valid_mask=valid_mask_eval, df_metrics=df_metrics)

        min_fhe = self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)
        min_mr = self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=df_metrics)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()

        print('PI: ', pi_eval)

        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = {eval_id[i]: (traj_eval[i], pi_eval[i])}
            else:
                self.test_predictions[data['scenario_id'][i]] = {eval_id[0]: (traj_eval[0], pi_eval[0])}
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        # save to csv
        df_metrics.to_csv('test_metrics.csv', mode='a', index=False, header=False)


    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            save_path = Path(self.submission_dir) / f'{self.submission_file_name}.parquet'
            ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            print('saved in: ', save_path)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser
