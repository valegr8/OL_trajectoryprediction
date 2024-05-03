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
import csv


import math
import numpy as np
from scipy.stats import norm
from scipy.stats import norm, expon, gamma, weibull_min, beta, pareto, lognorm

from torchviz import make_dot
from torch.autograd import Variable

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


from av2api import submission
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission


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
        print('NUM FUTURE STEPS: ', self.num_future_steps)
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


        print('LR: --------------',lr,type(lr))
        self.lr = 0.05
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

        self.save_loss_cls = []
        self.save_loss_refine = []
        self.save_loss_propose = []
        self.save_loss = [] 

        self.save_loss_grad = [] 

        self.optimizer = self.optimizer_MLP(self.lr)

        # Create an empty DataFrame
        self.df_metrics = pd.DataFrame(columns=['scenario_id','timestep','val_Brier','val_minADE','val_minAHE', 'val_minFDE','val_minFHE','val_minMR'])

        self.online_learning = True
        self.initial_hstorical_steps = 20
        self.ol_time_slice = 20
        self.final_his_step = 110
        self.dataset_steps = 110
        self.dataset_his_steps = 50
        self.ol_version = 0
        self.save_metrics_path = 'val_metrics.csv'

        self.p_values_list = [] 
        self.switch_on = 1

        self.alpha = 0.05 #used for selecting suspitious points
        self.gamma = 5
        self.lambda_val = 0.8

        # Get original weights for decoder modules
        self.original_weights = {}
        for module_name, module in self.named_modules():
            if 'decoder' in module_name:
                self.original_weights[module_name] = module.state_dict()


    def optimizer_MLP(self, lr):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name


                if ('decoder.to_loc_refine_pos.mlp' in full_param_name) or ('decoder.to_scale_refine_pos.mlp' in full_param_name) or ('decoder.to_pi.mlp' in full_param_name):
                    # print(full_param_name)
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
        # print('Updating optimizer - LEARNING RATE: ', lr)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=self.weight_decay)
        
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

    def set_initial_historical_step(self, initial_his_steps):
        self.initial_his_steps = initial_his_steps
        self.encoder.agent_encoder.set_initial_historical_steps(initial_his_steps)
        self.encoder.map_encoder.set_initial_historical_steps(initial_his_steps)
        self.decoder.set_initial_historical_steps(initial_his_steps)

    def compute_metrics(self, data, pred, num_gt_steps, online_learning = True):
        # Assign the tensor to the 'scenario_id' column
        # print(self.df_metrics['scenario_id'],'---',data['scenario_id'])
        # self.df_metrics['scenario_id'] = data['scenario_id']

        self.df_metrics = pd.DataFrame({'scenario_id': data['scenario_id']})
        self.df_metrics['timestep'] = self.num_historical_steps

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

        # the mask is needed only to calculate the loss of ceirtain elements, for stabilization, so that with the background thep we don update all weights
        cls_mask = data['agent']['predict_mask'][:, -1]
        if online_learning:
            # reduce size of traj propose and refine based on the number of steps we want to compare with the ground truth
            traj_propose = traj_propose[...,:num_gt_steps, :]
            traj_refine = traj_refine[...,:num_gt_steps, :]
            reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:self.num_historical_steps+num_gt_steps]
            reg_mask = torch.ones_like(reg_mask)
            cls_mask = torch.ones_like(cls_mask)

        else:
            reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]

        gt = torch.cat([data['agent']['target'][:,self.num_historical_steps:self.num_historical_steps+num_gt_steps, :self.output_dim], data['agent']['target'][:,self.num_historical_steps:self.num_historical_steps+num_gt_steps, -1:]], dim=-1)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

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

        loss = reg_loss_propose + reg_loss_refine + cls_loss

        if self.num_historical_steps % int(self.ol_time_slice/2) == 0:
            self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

            # self.save_loss_cls.append(cls_loss.cpu().detach().numpy())
            # self.save_loss_refine.append(reg_loss_refine.cpu().detach().numpy())
            # self.save_loss_propose.append(reg_loss_propose.cpu().detach().numpy())
            
            
            valid_mask_eval = reg_mask[eval_mask]
            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                        traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)


            gt_eval = gt[eval_mask]

            self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=self.df_metrics)
            self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=self.df_metrics)
            self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=self.df_metrics)
            self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,valid_mask=valid_mask_eval, df_metrics=self.df_metrics)
            self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=self.df_metrics)
            self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval, df_metrics=self.df_metrics)

            self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

            # print(self.df_metrics.iloc[-1])
            self.df_metrics.to_csv(self.save_metrics_path, mode='a', index=False, header=False)

            del reg_loss_propose
            del reg_loss_refine
            del cls_loss
        return loss


    def save_trajectory(self, data, pred):
        # save trajecotry to file submission
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        traj_eval= traj_eval.cpu().detach().numpy()
        pi_eval = pi_eval.cpu().detach().numpy()

        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    if self.online_learning:
                        self.test_predictions[data['scenario_id'][i],self.num_historical_steps] = {eval_id[i]: (traj_eval[i], pi_eval[i])}
                    else:
                        self.test_predictions[data['scenario_id'][i]] = {eval_id[i]: (traj_eval[i], pi_eval[i])}
            else:
                if self.online_learning:
                    self.test_predictions[data['scenario_id'][i],self.num_historical_steps] = {eval_id[0]: (traj_eval[0], pi_eval[0])}
                else:
                    self.test_predictions[data['scenario_id'][i]] = {eval_id[0]: (traj_eval[0], pi_eval[0])}


    # Function to plot histogram with fitted distributions
    def plot_with_fitted_distributions(self, ax, data, distributions, colors, labels):
        ax.hist(data, bins=20, density=True, alpha=0.6, color='gray', label='Data')
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        for distribution, color, label in zip(distributions, colors, labels):
            params = distribution.fit(data)
            pdf = distribution.pdf(x, *params)
            ax.plot(x, pdf, color=color, linewidth=2, label=label)
        ax.legend()

    def plot_losses(self):
        # Create subplots for each array
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Define distributions, colors, and labels
        distributions = [norm, expon, gamma, pareto, lognorm, beta]
        colors = ['b', 'r', 'g', 'y', 'c', 'm']
        labels = ['Normal', 'Exponential', 'Gamma', 'Pareto', 'Log-Normal', 'Beta']

        # Plot histograms with fitted distributions
        self.plot_with_fitted_distributions(axs[0, 0], self.save_loss_cls, distributions, colors, labels)
        self.plot_with_fitted_distributions(axs[0, 1], self.save_loss_refine, distributions, colors, labels)
        self.plot_with_fitted_distributions(axs[1, 0], self.save_loss_propose, distributions, colors, labels)
        self.plot_with_fitted_distributions(axs[1, 1], self.save_loss, distributions, colors, labels)


        # Set titles and labels
        axs[0, 0].set_title('Loss_cls')
        axs[0, 1].set_title('Loss_refine')
        axs[1, 0].set_title('Loss_propose')
        axs[1, 1].set_title('Total_loss')

        for ax in axs.flat:
            ax.set_xlabel('Loss Value')
            ax.set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def training_step(self,
                      data,
                      batch_idx,
                      num_gt_steps: int=60):
        
        #empty arrays that save the loss
        self.save_loss_cls = []
        self.save_loss_refine = []
        self.save_loss_propose = []
        self.save_loss = [] 

        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        if self.online_learning:
            loss = self.online_learning_update(data)
        else:
            pred = self(data)
            loss = self.compute_metrics(data, pred, num_gt_steps, self.online_learning)
            self.save_trajectory(data, pred)

        return loss

    def online_learning_update(self, data):
        torch.cuda.empty_cache()
        # Reset the model weights to their original values
        for module_name, module in self.named_modules():
            if 'decoder' in module_name:
                module.load_state_dict(self.original_weights[module_name])

        with torch.enable_grad():
            if self.switch_on == 0:
                for i in range(self.initial_hstorical_steps, self.final_his_step-self.ol_time_slice+1, int(self.ol_time_slice/2)):
                    self.set_num_historical_steps(i)

                    if self.ol_version == 0:
                        self.set_initial_historical_step(i-self.ol_time_slice)
                        num_gt_steps = self.final_his_step-self.num_historical_steps if self.final_his_step-self.num_historical_steps<self.num_future_steps else self.num_future_steps
                    else:
                        num_gt_steps = self.ol_time_slice

                    pred = self(data)
                    loss = self.compute_metrics(data, pred, num_gt_steps, self.online_learning)
                    self.save_trajectory(data, pred)
                    # print(i,loss)

                    if i < self.final_his_step-self.ol_time_slice:
                        # clear gradients
                        self.optimizer.zero_grad()

                        a = {}
                        b = {}

                        # for module_name, module in self.named_modules():
                        #     for param_name, param in module.named_parameters():
                        #         full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                        #         if (('decoder.to_loc_refine_pos.mlp' in full_param_name) or ('decoder.to_scale_refine_pos.mlp' in full_param_name) or ('decoder.to_pi.mlp' in full_param_name)):
                        #             a[full_param_name] = list(param)[0].clone()

                        loss.backward()

                        # Update model parameters
                        self.optimizer.step()

                        # for module_name, module in self.named_modules():
                        #     if module.training:
                        #         print("Model is in training mode")
                        #         make_dot(loss,params=dict(module.named_parameters()), show_attrs=True, show_saved=True).render("computational_graph", format="png")

                        #     else:
                        #         print("Model is in evaluation mode")

                        # Access the weights of the model
                        # for module_name, module in self.named_modules():
                        #     for param_name, param in module.named_parameters():
                        #         full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                        #         if (('decoder.to_loc_refine_pos.mlp' in full_param_name) or ('decoder.to_scale_refine_pos.mlp' in full_param_name) or ('decoder.to_pi.mlp' in full_param_name)):
                        #             b[full_param_name] = list(param)[0].clone()
                        #             print(full_param_name,'equal? ',torch.equal(a[full_param_name].data, b[full_param_name].data))
                                    # param.data,a[full_param_name].data
                                    # print(full_param_name,list(param)[0].requires_grad, list(param)[0].grad, list(param)[0].is_leaf)

                self.set_num_historical_steps(self.final_his_step)

                return loss
            
            elif self.switch_on == 1:
                # Clear the memory used by the GPU
                torch.cuda.empty_cache()

                # reset optimizer to original learning rate
                # self.optimizer = self.optimizer_MLP(self.lr)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr


                #compute loss at each timestep
                for i in range(self.initial_hstorical_steps, self.final_his_step-self.ol_time_slice+1, 1):
                    torch.cuda.empty_cache()
                    normal_list = []  
                    susp_difference_value_drifts = [] 
                    norm_diff_value_drifts = []  


                    self.set_num_historical_steps(i)

                    if self.ol_version == 0:
                        self.set_initial_historical_step(i-self.ol_time_slice)
                        num_gt_steps = self.final_his_step-self.num_historical_steps if self.final_his_step-self.num_historical_steps<self.num_future_steps else self.num_future_steps
                    else:
                        num_gt_steps = self.ol_time_slice

                    if (i % int(self.ol_time_slice/2) == 0):
                        pred = self(data)
                    else:
                        with torch.no_grad():
                            pred = self(data)


                    loss = self.compute_metrics(data, pred, num_gt_steps, self.online_learning)

                    del pred
                    torch.cuda.empty_cache()

                    
                    self.save_loss.append(loss.cpu().detach().numpy())
                    # self.save_trajectory(data, pred)
                    # print(i,loss)

                    # self.save_loss_grad.append(loss)
                    if (i < self.final_his_step-self.ol_time_slice) and (i % int(self.ol_time_slice/2) == 0):
                        # self.save_trajectory(data, pred)
                    #pvalue update learning rate
                    # if i % int(self.ol_time_slice/2) == 0:
                        if not (i - self.ol_time_slice) == 0:
                            # print('IIII',i)
                            # Example usage:
                            # Displaying results
                            mu = np.mean(self.save_loss)  # Mean of losses
                            sigma = np.std(self.save_loss)  # Standard deviation of losses
                            for j, no_grad_loss in enumerate(self.save_loss):
                                with open('/home/vgrwbx/workspace/OL_trajectoryprediction/losses.csv', 'a', newline='') as losses_file:
                                    writer_losses = csv.writer(losses_file)
                                    writer_losses.writerow([data['scenario_id'][0], i-int(self.ol_time_slice/2)+j, no_grad_loss])

                                with open('/home/vgrwbx/workspace/OL_trajectoryprediction/velocities.csv', 'a', newline='') as vel_file:
                                    writer_losses = csv.writer(vel_file)
                                    writer_losses.writerow([data['scenario_id'][0], i-int(self.ol_time_slice/2)+j, data['agent']['velocity'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j-1, i-int(self.ol_time_slice/2)+j-1]])



                                z_score = (no_grad_loss - mu) / sigma
            
                                # Compute the p-value using the CDF of standard normal distribution
                                p_value = norm.cdf(z_score)

                                self.p_values_list.append([data['scenario_id'][0], i-int(self.ol_time_slice/2)+j, p_value])
                                                                                           
       
                                # print(f"Loss: {no_grad_loss}, p-value: {p_value}")
                                if p_value <= self.alpha or p_value >= 1 - self.alpha:
                                    # print(j,'suspitious', p_value, data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                                    # suspitious_list.append(data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                                    diff_val_drift = [] 
                                    if not (j == 0):
                                        lower_dist = self.calculate_distance(data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim], data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j-1, :self.input_dim])
                                        diff_val_drift.append(lower_dist)
                                    if not (j == int(self.ol_time_slice/2)):
                                        upper_dist = self.calculate_distance(data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim], data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j+1, :self.input_dim])
                                        diff_val_drift.append(upper_dist)

                                    susp_difference_value_drifts.append(self.calculate_mean(diff_val_drift))
                                else:
                                    # print(j,'normal', p_value, data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                                    normal_list.append(data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                            
                            for k, n_point in enumerate(normal_list):
                                # print(j, p_value, data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                                # suspitious_list.append(data['agent']['position'][data['agent']['category'] == 3, i-int(self.ol_time_slice/2)+j, :self.input_dim])
                                diff_val_drift = [] 
                                if not (k == 0):
                                    lower_dist = self.calculate_distance(n_point, normal_list[k-1])
                                    diff_val_drift.append(lower_dist)
                                if not (k == len(normal_list)-1):
                                    upper_dist = self.calculate_distance(n_point, normal_list[k+1])
                                    diff_val_drift.append(upper_dist)

                                norm_diff_value_drifts.append(self.calculate_mean(diff_val_drift))
                            
                            if (len(susp_difference_value_drifts) == 0 or (len(susp_difference_value_drifts) == 1 and susp_difference_value_drifts[0] == None)):  
                                diff_val_drift = self.calculate_mean(norm_diff_value_drifts)
                            elif (len(norm_diff_value_drifts) == 0 or (len(norm_diff_value_drifts) == 1 and norm_diff_value_drifts[0] == None)): 
                                diff_val_drift = self.calculate_mean(susp_difference_value_drifts)
                            else:
                                diff_val_drift = self.calculate_mean(norm_diff_value_drifts) * self.calculate_mean(susp_difference_value_drifts)

                            suspitious_rate = len(susp_difference_value_drifts)/len(self.save_loss)

                            # print(i, 'diff_val_drift', diff_val_drift, 'suspitious rate',suspitious_rate)

                            if diff_val_drift < self.gamma:
                                # Calculate the weight using the exponential decay term
                                weight = self.lambda_val * math.exp(-diff_val_drift) + (1 - self.lambda_val) * suspitious_rate
                            else:
                                # Use the suspicion ratio directly
                                weight = self.lambda_val + (1 - self.lambda_val) * suspitious_rate
                            
                            new_lr = self.lr * weight
                                
                            # reset optimier
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr

                        # reset loss array       
                        self.save_loss = []
                        
                        # clear gradients
                        self.optimizer.zero_grad()
                        loss.backward()
                        # Update model parameters
                        self.optimizer.step()
                        
                        # Clear the memory used by the GPU
                        del loss
                        torch.cuda.empty_cache()

                return loss
        
    def calculate_mean(self, list):
        if len(list) == 0:
            return None  # Return None for an empty list (or handle as needed)
        total = sum(list)
        mean = total / len(list)
        return mean

    def calculate_distance(self, p1, p2):
        # Ensure that p1 and p2 have the same shape
        assert p1.shape == p2.shape, "Tensors must have the same shape"

        # Compute the element-wise squared differences
        squared_diff = (p1 - p2)**2

        # Sum along all dimensions except the first (batch) dimension
        sum_squared_diff = torch.sum(squared_diff, dim=tuple(range(1, len(p1.shape))))

        # Compute the square root of the summed squared differences
        distance = torch.sqrt(sum_squared_diff)

        return distance
        
    def on_train_end(self):
        # Define column names for the DataFrame
        columns = ['scenario_id', 'timestep', 'p_value']

        # Create DataFrame from self.p_values_list
        self.df_pvalues = pd.DataFrame(self.p_values_list, columns=columns)
        # Save the DataFrame to a CSV file
        csv_filename = 'pvalues.csv'
        self.df_pvalues.to_csv(csv_filename, index=False)  # Specify index=False to exclude row indices in the CSV file

        print("DataFrame saved to CSV file:", csv_filename)
        # self.plot_losses()

        if self.dataset == 'argoverse_v2':
            save_path = Path(self.submission_dir) / f'{self.submission_file_name}.parquet'
            if self.online_learning:
                submission.ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            else:
                ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            print('saved in: ', save_path)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        print('Metrics saved to ', self.save_metrics_path)


    def validation_step(self,
                        data,
                        batch_idx):
        # Clear the memory used by the GPU
        torch.cuda.empty_cache()

        # if self.online_learning:
            # for module_name, module in self.named_modules():
                # if 'decoder' in module_name:
                        # Reset the model weights to their original values

                        # module.load_state_dict(self.original_weights[module_name])
                # for param_name, param in module.named_parameters():
                #     full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                #     if ('decoder.to_loc_refine_pos.mlp' in full_param_name) or ('decoder.to_scale_refine_pos.mlp' in full_param_name) or ('decoder.to_pi.mlp' in full_param_name):
                #         param.requires_grad_(True)
                #     else:
                #         param.requires_grad_(False)

        print('validation-------------------------------------------')

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        pred = self(data)

        if self.dataset == 'argoverse_v2' and self.final_his_step <= self.dataset_steps - self.num_future_steps:
            self.compute_metrics(data, pred, num_gt_steps=self.num_future_steps, online_learning=False)

        self.save_trajectory(data, pred)

    def on_validation_end(self):
        if self.dataset == 'argoverse_v2':
            save_path = Path(self.submission_dir) / f'{self.submission_file_name}_val.parquet'

            if self.online_learning:
                submission.ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            else:
                ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            print('saved in: ', save_path)
            print('Metrics saved to ', self.save_metrics_path)

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

        self.save_trajectory(data, pred)

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            save_path = Path(self.submission_dir) / f'{self.submission_file_name}.parquet'
            if self.online_learning:
                submission.ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            else:
                ChallengeSubmission(self.test_predictions).to_parquet(save_path)
            print('saved in: ', save_path)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        print('Metrics saved to ', self.save_metrics_path)

    def configure_optimizers(self):
        if self.online_learning:
            return self.optimizer_MLP(self.lr)
        else:
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

            print('configure optimizer--------------------------------------------------------------------------------')

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
