import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn

from networks import Generator, Discriminator, DeepLabV2_ResNet101_local_MSC
from losses import init_loss
from tools import resize_target, MeaninglessError

class OurModel(nn.Module):
    def __init__(self, hp, class_emb_vis, class_emb_all):
        super(OurModel, self).__init__()
        self.hp = hp

        self.Em_vis = nn.Embedding.from_pretrained(class_emb_vis).cuda()
        self.Em_vis.weight.requires_grad = False
        self.Em_all = nn.Embedding.from_pretrained(class_emb_all).cuda()
        self.Em_all.weight.requires_grad = False

        self.prior = np.ones((hp['dis']['out_dim_cls']-1))
        for k in range(hp['dis']['out_dim_cls']-hp['num_unseen']-1, hp['dis']['out_dim_cls']-1):
            self.prior[k] = self.prior[k] + hp['gen_unseen_rate']
        self.prior_ = self.prior / np.linalg.norm(self.prior, ord=1)

        self.gen = Generator(hp['gen'])
        self.dis = Discriminator(hp['dis'])
        self.back = DeepLabV2_ResNet101_local_MSC(hp['back'])

        self.discLoss, self.contentLoss, self.clsLoss = init_loss(hp)

    def forward(self, data, gt, mode):
        assert (mode == 'step1' or mode == 'step2')
        self.init_all(mode)
        flag = 1

        try:
            ignore_mask = (gt != self.hp['ignore_index']).cuda()
            if not (ignore_mask.sum() > 0): # meaningless batch
                raise MeaninglessError()

            if mode == 'step1': # step1
                self.set_mode('step1')
    
                self.loss_KLD, self.target_all, self.target, self.contextual = self.back(data, ignore_mask)
                self.target_shape_all = [x.shape for x in self.target_all]
                self.gt_all = [resize_target(gt, x[2]).cuda() for x in self.target_shape_all]
                self.ignore_mask_all = [(x != self.hp['ignore_index']).cuda() for x in self.gt_all]
                if not all([x.sum() > 0 for x in self.ignore_mask_all]): # meaningless batch
                    raise MeaninglessError()

                # self.target_shape = self.target.shape
                # self.contextual_shape = self.contextual.shape
                self.gt = resize_target(gt, self.target.shape[2]).cuda()
                self.ignore_mask = (self.gt != self.hp['ignore_index']).cuda()
                if not (self.ignore_mask.sum() > 0): # meaningless batch
                    raise MeaninglessError()

                condition = self.Em_vis(self.gt).permute(0,3,1,2).contiguous()
                self.sample = torch.cat((condition, self.contextual), dim=1)
                self.predict = self.gen(self.sample.detach())

            else:               # step2
                self.set_mode('step2')
                
                with torch.no_grad():
                    _, _, self.target, self.contextual = self.back(data, ignore_mask)
                    self.target_shape = self.target.shape
                    self.contextual_shape = self.contextual.shape

                self.gt = torch.LongTensor(
                            np.random.choice(
                                #a=range(self.Em_all.shape[0]), 
                                a=range(self.hp['dis']['out_dim_cls']-1),
                                size=(self.target_shape[0], self.target_shape[2], self.target_shape[3]), 
                                replace=True,
                                p=self.prior_
                            )
                          ).cuda()
                self.ignore_mask = (self.gt != self.hp['ignore_index']).cuda()
                if not (self.ignore_mask.sum() > 0): # meaningless batch
                    raise MeaninglessError()

                condition = self.Em_all(self.gt).permute(0,3,1,2).contiguous()
                random_noise = torch.randn(self.contextual_shape).cuda()
                self.sample = torch.cat((condition, random_noise), dim=1)
                self.predict = self.gen(self.sample.detach())       

        except MeaninglessError:
            flag = -1

        assert (flag == 1 or flag ==-1)
        if flag == 1:
            self.get_loss_D(mode)
            if self.hp['update_back'] == 't':
                self.get_loss_B()
            self.get_loss_G(mode)

        return self.get_losses(flag, mode)

    def test(self, data, gt):
        with torch.no_grad():
            self.set_mode('test')

            flag = 1
            try:
                ignore_mask = (gt != self.hp['ignore_index']).cuda()
                _, _, self.target, _ = self.back(data, ignore_mask)
                self.gt = resize_target(gt, self.target.shape[2]).cuda()
                self.ignore_mask = (self.gt != self.hp['ignore_index']).cuda()
                if not (self.ignore_mask.sum() > 0): # meaningless batch
                    raise MeaninglessError()
            except MeaninglessError:
                flag = -1

            assert (flag == 1 or flag ==-1)
            if flag == 1:
                self.get_loss_D('test')

            return self.get_losses(flag, 'test')

    def get_loss_D(self, mode):
        assert (mode == 'step1' or mode == 'step2' or mode == 'test')
        if mode == 'step1':
            self.loss_D_GAN, self.loss_D_real, self.loss_D_fake, self.loss_D_gp = \
                        self.discLoss(self.dis, self.predict.detach(), self.target.detach(), self.ignore_mask)
            self.loss_cls_fake, self.acc_cls_fake, _, _ = self.clsLoss(self.dis, self.predict, self.gt, self.ignore_mask)
            for (target,  gt, ignore_mask) in zip(self.target_all,  self.gt_all, self.ignore_mask_all):    
                loss_cls_real, acc_cls_real, _, _ = self.clsLoss(self.dis, target, gt, ignore_mask) # backward to backbone, no detach
                self.loss_cls_real += loss_cls_real
                self.acc_cls_real += acc_cls_real
            total = len(self.target_all)
            self.loss_cls_real /= total
            self.acc_cls_real /= total
            self.loss_D_cls_fake = self.loss_cls_fake * self.hp['lambda_D_cls_fake']
            self.loss_D_cls_real = self.loss_cls_real * self.hp['lambda_D_cls_real']
            self.loss_D_cls = self.loss_D_cls_fake + self.loss_D_cls_real
            self.loss_D = self.loss_D_GAN + self.loss_D_cls
        elif mode == 'step2':
            self.loss_cls_fake, self.acc_cls_fake, _, _ = self.clsLoss(self.dis, self.predict, self.gt, self.ignore_mask) # backward to generator, no detach
            self.loss_D_cls_fake = self.loss_cls_fake * self.hp['lambda_D_cls_fake_transfer']
            self.loss_D_cls = self.loss_D_cls_fake
            self.loss_D = self.loss_D_cls
        else:
            with torch.no_grad():
                _, _, self.pred_cls_real, self.sorted_indices = self.clsLoss(self.dis, self.target, self.gt, self.ignore_mask)

    def get_loss_G(self, mode):
        assert (mode == 'step1' or mode == 'step2')
        if mode == 'step1':
            self.loss_G_GAN = self.discLoss.get_g_loss(self.dis, self.predict, self.ignore_mask)
            loss_G_Content = self.contentLoss(self.predict, self.target.detach(), self.gt, self.ignore_mask)
            self.loss_G_Content = loss_G_Content * self.hp['lambda_G_Content']
            self.loss_G_cls = self.loss_cls_fake * self.hp['lambda_G_cls']
            self.loss_G = self.loss_G_GAN * self.hp['lambda_G_GAN'] + self.loss_G_Content + self.loss_G_cls
        else:
            self.loss_G_cls = self.loss_cls_fake * self.hp['lambda_G_cls_transfer']
            self.loss_G = self.loss_G_cls

    def get_loss_B(self):
        self.loss_B_KLD = self.loss_KLD * self.hp['lambda_B_KLD']
        self.loss_B_cls = self.loss_cls_real * self.hp['lambda_B_cls']
        self.loss_B = self.loss_B_KLD + self.loss_B_cls

    def set_mode(self, mode):
        assert (mode == 'step1' or mode == 'step2' or mode == 'test')
        if mode == 'step1':
            self.train()
            self.back.freeze_bn()
        elif mode == 'step2':
            self.train()
            self.back.eval()
        else:
            self.eval()
            self.dis.eval()
            self.back.eval()
            self.gen.eval()
        self.Em_vis.eval()
        self.Em_all.eval()

    def init_all(self, mode):
        assert (mode == 'step1' or mode == 'step2')
        if mode == 'step1':
            self.loss_G_GAN = 0
            self.loss_G_Content = 0
            self.loss_G_cls = 0
            self.loss_G = 0
            self.loss_B_KLD = 0
            self.loss_B_cls = 0
            self.loss_B = 0
            self.loss_D_real = 0
            self.loss_D_fake = 0
            self.loss_D_gp = 0
            self.loss_D_GAN = 0
            self.loss_D_cls_real = 0
            self.loss_D_cls_fake = 0
            self.loss_D_cls = 0
            self.loss_D = 0
            self.loss_cls_real = 0
            self.loss_cls_fake = 0
            self.acc_cls_real = 0
            self.acc_cls_fake = 0
        else:
            self.loss_G_cls = 0
            self.loss_G = 0
            self.loss_D_cls_fake = 0
            self.loss_D_cls = 0
            self.loss_D = 0
            self.loss_cls_fake = 0
            self.acc_cls_fake = 0

    def get_losses(self, flag, mode):
        assert (mode == 'step1' or mode == 'step2' or mode == 'test')
        zero_tensor = torch.from_numpy(np.array(0)).cuda()

        if mode == 'step1':
            if flag == 1:
                return torch.from_numpy(np.array(flag)).long().cuda(),\
                       self.loss_G_GAN,\
                       self.loss_G_Content,\
                       self.loss_G_cls,\
                       self.loss_G,\
                       self.loss_B_KLD if self.hp['update_back'] == 't' else zero_tensor,\
                       self.loss_B_cls if self.hp['update_back'] == 't' else zero_tensor,\
                       self.loss_B if self.hp['update_back'] == 't' else zero_tensor,\
                       self.loss_D_real,\
                       self.loss_D_fake,\
                       self.loss_D_gp if self.loss_D_gp != None else zero_tensor,\
                       self.loss_D_GAN,\
                       self.loss_D_cls_real,\
                       self.loss_D_cls_fake,\
                       self.loss_D_cls,\
                       self.loss_D,\
                       self.loss_cls_real,\
                       self.loss_cls_fake,\
                       self.acc_cls_real,\
                       self.acc_cls_fake
            else:   
                return torch.from_numpy(np.array(flag)).long().cuda(),\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor
        elif mode == 'step2':
            if flag == 1:
                return torch.from_numpy(np.array(flag)).long().cuda(),\
                       self.loss_G_cls,\
                       self.loss_G,\
                       self.loss_D_cls_fake,\
                       self.loss_D_cls,\
                       self.loss_D,\
                       self.loss_cls_fake,\
                       self.acc_cls_fake
            else:
                return torch.from_numpy(np.array(flag)).long().cuda(),\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor,\
                       zero_tensor             
        else:
            with torch.no_grad():
                if flag == 1:
                    return torch.from_numpy(np.array(flag)).long().cuda(),\
                           self.pred_cls_real,\
                           self.sorted_indices,\
                           self.gt,\
                           self.ignore_mask  # original label and corresponding ignore mask 
                else:   
                    return torch.from_numpy(np.array(flag)).long().cuda(),\
                           zero_tensor,\
                           zero_tensor,\
                           zero_tensor,\
                           zero_tensor
