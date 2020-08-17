"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from tools import MeaninglessError
from model import OurModel

class Trainer(nn.Module):
    def __init__(self, cfg, class_emb_vis, class_emb_all, schedule, checkpoint_dir=None, resume_from=None):
        super(Trainer, self).__init__()
        assert (schedule in ['step1', 'mixed', 'st', 'st_mixed'])

        self.cfg = cfg
        self.schedule = schedule
        self.model = OurModel(cfg, class_emb_vis, class_emb_all)

        if self.schedule == 'step1':
            lr_name = 'lr'
        elif self.schedule == 'mixed':
            lr_name = 'lr_transfer'
        elif self.schedule == 'st':
            lr_name = 'lr_st'
        else:
            lr_name = 'lr_st_transfer'

        """
        Optimizer for backbone
        """
        self.back_opt = {
            "sgd": torch.optim.SGD(
                params=[
                    {
                        "params": get_params(self.model.back, key="1x"),
                        "lr": cfg['back_opt'][lr_name],
                        "initial_lr": cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="5x"),
                        "lr": 5 * cfg['back_opt'][lr_name],
                        "initial_lr": 5 * cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="10x"),
                        "lr": 10 * cfg['back_opt'][lr_name],
                        "initial_lr": 10 * cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="20x"),
                        "lr": 20 * cfg['back_opt'][lr_name],
                        "initial_lr": 20 * cfg['back_opt'][lr_name],
                        "weight_decay": 0.0,
                    }
                ],
                momentum=cfg['back_opt']['MOMENTUM'],
            ),
            "adam": torch.optim.Adam(
                params=[
                    {
                        "params": get_params(self.model.back, key="1x"),
                        "lr": cfg['back_opt'][lr_name],
                        "initial_lr": cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="5x"),
                        "lr": 5 * cfg['back_opt'][lr_name],
                        "initial_lr": 5 * cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="10x"),
                        "lr": 10 * cfg['back_opt'][lr_name],
                        "initial_lr": 10 * cfg['back_opt'][lr_name],
                        "weight_decay": cfg['back_opt']['WEIGHT_DECAY'],
                    },
                    {
                        "params": get_params(self.model.back, key="20x"),
                        "lr": 20 * cfg['back_opt'][lr_name],
                        "initial_lr": 20 * cfg['back_opt'][lr_name],
                        "weight_decay": 0.0,
                    }
                ]
            )
        }.get(cfg['back_opt']['OPTIMIZER'])

        """
        Optimizer for discriminator
        """
        self.dis_opt = {
            'RMSprop': torch.optim.RMSprop(
                params=[
                    {
                        'params': [p for p in list(self.model.dis.parameters()) if p.requires_grad],
                        'initial_lr': cfg['dis_opt'][lr_name]
                    }
                ],
                lr=cfg['dis_opt'][lr_name], 
                weight_decay=cfg['dis_opt']['weight_decay']
            ),
            'adam': torch.optim.Adam(
                params=[
                    {
                        'params': [p for p in list(self.model.dis.parameters()) if p.requires_grad],
                        'initial_lr': cfg['dis_opt'][lr_name]
                    }
                ],
                lr=cfg['dis_opt'][lr_name], 
                weight_decay=cfg['dis_opt']['weight_decay']
            )
        }.get(cfg['dis_opt']['OPTIMIZER'])

        """
        Optimizer for generator
        """
        self.gen_opt = {
            'RMSprop': torch.optim.RMSprop(
                params=[
                    {
                        'params': [p for p in list(self.model.gen.parameters()) if p.requires_grad],
                        'initial_lr': cfg['gen_opt'][lr_name]
                    }
                ],
                lr=cfg['gen_opt'][lr_name], 
                weight_decay=cfg['gen_opt']['weight_decay']
            ),
            'adam': torch.optim.Adam(
                params=[
                    {
                        'params': [p for p in list(self.model.gen.parameters()) if p.requires_grad],
                        'initial_lr': cfg['gen_opt'][lr_name]
                    }
                ],
                lr=cfg['gen_opt'][lr_name], 
                weight_decay=cfg['gen_opt']['weight_decay']
            )
        }.get(cfg['gen_opt']['OPTIMIZER'])

        if resume_from >= 1:
            self.resume(checkpoint_dir, resume_from)
        else:
            self.from_scratch()

    def train(self, data, gt, mode, multigpus):
        assert (mode == 'step1' or mode == 'step2')

        # if mode == 'step1':
        #     this_model.back.train()
        #     this_model.back.freeze_bn()
        # elif mode == 'step2':
        #     this_model.back.eval()
        # else:
        #     raise NotImplementedError('Mode {} not supported.' % mode)
 
        # this_model.init_all(mode)        

        if mode == 'step1':
            flag, loss_G_GAN, loss_G_Content, loss_G_cls, loss_G, loss_B_KLD, loss_B_cls, loss_B, loss_D_real, loss_D_fake, loss_D_gp, loss_D_GAN, loss_D_cls_real, loss_D_cls_fake, loss_D_cls, loss_D, loss_cls_real, loss_cls_fake, acc_cls_real, acc_cls_fake = self.model(data, gt, mode)
        else:
            flag, loss_G_cls, loss_G, loss_D_cls_fake, loss_D_cls, loss_D, loss_cls_fake, acc_cls_fake = self.model(data, gt, mode)
        if (flag == -1).any() == True:
            raise MeaninglessError()

        if mode == 'step1':
            for iter_d in range(self.cfg['criticUpdates']):
                self.dis_opt.zero_grad()
                if iter_d == 0:
                    loss_D.mean().backward(retain_graph=True)
                else:
                    loss_D_GAN.mean().backward(retain_graph=True)
                self.dis_opt.step()
        else:
            self.dis_opt.zero_grad()
            loss_D.mean().backward(retain_graph=True)
            self.dis_opt.step()
        if self.dis_scheduler != None:
            self.dis_scheduler.step()

        if self.cfg['update_back'] == 't':
            if mode == 'step1':
                self.back_opt.zero_grad()
                loss_B.mean().backward()
                self.back_opt.step()
            if self.back_scheduler != None:
                self.back_scheduler.step()

        self.gen_opt.zero_grad()
        loss_G.mean().backward()
        self.gen_opt.step()
        if self.gen_scheduler != None:
            self.gen_scheduler.step()

        if mode == 'step1':
            return {
                'loss_G_GAN': loss_G_GAN.mean().item(),
                'loss_G_Content': loss_G_Content.mean().item(),
                'loss_G_cls': loss_G_cls.mean().item(),
                'loss_G': loss_G.mean().item(),
                'loss_B_KLD': loss_B_KLD.mean().item() if self.cfg['update_back'] == 't' else 0,
                'loss_B_cls': loss_B_cls.mean().item() if self.cfg['update_back'] == 't' else 0,
                'loss_B': loss_B.mean().item() if self.cfg['update_back'] == 't' else 0,
                'loss_D_real': loss_D_real.mean().item(),
                'loss_D_fake': loss_D_fake.mean().item(),
                'loss_D_gp': loss_D_gp.mean().item() if self.cfg['gan_type'] == 'wgan-gp' else None,
                'loss_D_GAN': loss_D_GAN.mean().item(),
                'loss_D_cls_real': loss_D_cls_real.mean().item(),
                'loss_D_cls_fake': loss_D_cls_fake.mean().item(),
                'loss_D_cls': loss_D_cls.mean().item(),
                'loss_D': loss_D.mean().item(),
                'loss_cls_real': loss_cls_real.mean().item(),
                'loss_cls_fake': loss_cls_fake.mean().item(),
                'acc_cls_real': acc_cls_real.mean().item(),
                'acc_cls_fake': acc_cls_fake.mean().item()
            }
        else:
            return {
                'loss_G_cls': loss_G_cls.mean().item(),
                'loss_G': loss_G.mean().item(),
                'loss_D_cls_fake': loss_D_cls_fake.mean().item(),
                'loss_D_cls': loss_D_cls.mean().item(),
                'loss_D': loss_D.mean().item(),
                'loss_cls_fake': loss_cls_fake.mean().item(),
                'acc_cls_fake': acc_cls_fake.mean().item()
            }

    def test(self, data, gt, multigpus):
        with torch.no_grad():
            this_model = self.model.module if multigpus else self.model

            flag, pred_cls_real, sorted_indices, resized_gt, resized_ignore_mask = this_model.test(data, gt)
            if (flag == -1).any() == True:
                raise MeaninglessError()

            return {
                'pred_cls_real': pred_cls_real,
                'sorted_indices': sorted_indices,
                'resized_gt': resized_gt,  # original label without mapping
                'resized_ignore_mask': resized_ignore_mask  # ignore mask w.r.t original label
            }

    def get_lr(self):
        return {
            'dis_lr': self.dis_opt.param_groups[0]["lr"],
            'gen_lr': self.gen_opt.param_groups[0]["lr"],
            'back_lr': self.back_opt.param_groups[0]["lr"]
        }

    def from_scratch(self):
        """
        Weight initialization
        """
        self.apply(weights_init(self.cfg['init']))

        if self.cfg['init_model'] != 'none':
            state_dict = torch.load(self.cfg['init_model'])
            if 'state_dict' in state_dict.keys():
                self.model.load_state_dict(state_dict['state_dict'])
            else:
                self.model.load_state_dict(state_dict, strict=False)

        """
        lr scheduler
        """
        self.back_scheduler = get_scheduler(self.back_opt, self.cfg['back_scheduler'], self.schedule)
        self.dis_scheduler = get_scheduler(self.dis_opt, self.cfg['dis_scheduler'], self.schedule)
        self.gen_scheduler = get_scheduler(self.gen_opt, self.cfg['gen_scheduler'], self.schedule)

    def resume(self, checkpoint_dir, resume_from):
        """
        Weight initialization
        """
        last_model = os.path.join(checkpoint_dir, '%08d.pth' % resume_from)
        state_dict = torch.load(last_model)
        self.model.load_state_dict(state_dict['state_dict'])

        """
        lr scheduler
        """
        self.dis_scheduler = get_scheduler(self.dis_opt, self.cfg['dis_scheduler'], self.schedule, resume_from)
        self.gen_scheduler = get_scheduler(self.gen_opt, self.cfg['gen_scheduler'], self.schedule, resume_from)
        self.back_scheduler = get_scheduler(self.back_opt, self.cfg['back_scheduler'], self.schedule, resume_from)

    def save(self, snapshot_dir, iterations, multigpus):
        this_model = self.model.module if multigpus else self.model

        model_name = os.path.join(snapshot_dir, '%08d.pth' % (iterations + 1))
        torch.save(
            {
                'iteration': iterations + 1,
                'state_dict': this_model.state_dict()
            },
            model_name
        )

    def forward(self, *inputs):
        raise NotImplementedError('Forward function not implemented.')

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0] or "vgg" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For contextual module
    if key == "5x":
        for m in model.named_modules():
            if "contextual" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias

def get_scheduler(optimizer, hp, step_schedule, it=-1):
    assert (step_schedule in ['step1', 'mixed', 'st', 'st_mixed'])
    if hp['lr_policy'] == 'constant':
        scheduler = None
    elif hp['lr_policy'] == 'poly':
        if step_schedule == 'step1':
            init_lr_ = hp['init_lr']
            max_epoch_ = hp['max_iter']
        elif step_schedule == 'mixed':
            init_lr_ = hp['init_lr_transfer']
            max_epoch_ = hp['max_iter_transfer']
        elif step_schedule == 'st':
            init_lr_ = hp['init_lr_st']
            max_epoch_ = hp['max_iter_st']
        else:
            init_lr_ = hp['init_lr_st_transfer']
            max_epoch_ = hp['max_iter_st_transfer']
        scheduler = poly_lr_scheduler(
            optimizer, 
            init_lr=init_lr_, 
            lr_decay_epoch=hp['lr_decay_iter'], 
            power=hp['power'], 
            max_epoch=max_epoch_, 
            last_epoch=it
        )
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=hp['step_size'], 
            gamma=hp['gamma'], 
            last_epoch=it
        )
    elif hp['lr_policy'] == 'lambda':
        lambda_G = lambda epoch: 1.0 if epoch < hp['start_decay_iter'] else \
                       hp['gamma'] ** ((epoch - hp['start_decay_iter']) // hp['step_size'] + 1)
        scheduler = lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda_G, 
            last_epoch=it
        )
    else:
        raise NotImplementedError('%s not implemented', hp['lr_policy'])

    return scheduler


class poly_lr_scheduler():
    def __init__(self, optimizer, init_lr, lr_decay_epoch, power, max_epoch, last_epoch=-1):
        assert (last_epoch >= -1 and last_epoch != 0)
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.power = power
        self.max_epoch = max_epoch
        self.now_epoch = 0 if last_epoch == -1 else last_epoch
        self.update()

    def update(self):  
        new_lr = self.init_lr * (1 - float(self.now_epoch) / self.max_epoch) ** self.power
        self.optimizer.param_groups[0]["lr"] = new_lr
        self.optimizer.param_groups[1]["lr"] = 5 * new_lr
        self.optimizer.param_groups[2]["lr"] = 10 * new_lr
        self.optimizer.param_groups[3]["lr"] = 20 * new_lr

    def step(self):
        self.now_epoch += 1
        if self.now_epoch % self.lr_decay_epoch == 0 and self.now_epoch < self.max_epoch:
            self.update()
