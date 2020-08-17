import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models
import torch.nn.functional as F
from tools import resize_target

###############################################################################
# Loss Functions
###############################################################################

class ContentLoss(nn.Module):
    def __init__(self, loss):
        super(ContentLoss, self).__init__()
        self.criterion = loss

    def forward(self, predict, target, gt, ignore_mask):
        mask = ignore_mask.float().unsqueeze(1).repeat(1, predict.shape[1], 1, 1)
        return (self.criterion(predict, target) * mask).sum() / mask.sum()

class PerceptualLoss(nn.Module):
    def __init__(self, loss):
        super(PerceptualLoss, self).__init__()
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features.cuda()
        model = nn.Sequential().cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, predict, target, gt, ignore_mask):
        f_predict = self.contentFunc.forward(predict)
        f_target = self.contentFunc.forward(target).detach()
        mask = ignore_mask.float().unsqueeze(1).repeat(1, f_predict.shape[1], 1, 1)
        return (self.criterion(f_predict, f_target) * mask).sum() / mask.sum()

class MMDLoss(nn.Module):
    def __init__(self, opt, use_sqrt=True, sigma=[2,5,10,20,40,60]):
        super(MMDLoss, self).__init__()
        self.ignore = opt['ignore_index']
        self.use_sqrt = use_sqrt
        self.sigma = sigma

    """
    M: Number of predict samples
    N: Number of target samples
    """
    def get_scale_matrix(self, M, N):
        s = torch.zeros((M + N, 2)).cuda()
        s[:M, 0] = 1.0 / M
        s[M:, 1] = 1.0 / N
        s -= 1.0 / (M + N)
        return s

    """
    Calculates cost of the network, which is square root of the mixture of 'K' RBF kernels.
    sigma: Bandwidth parameters for the 'K' kernels.
    """
    def cal_loss(self, predict, target):
        M, N = predict.shape[0], target.shape[0]
        W = self.get_scale_matrix(M, N)
        ww = torch.matmul(W, W.t())

        X = torch.cat((predict, target), 0)
        XX = torch.matmul(X, X.t())
        x = torch.diag(XX, 0).repeat(M + N, 1)
        prob_mat = XX - 0.5 * x - 0.5 * x.t()
        
        loss = 0.0
        for bw in self.sigma:
            K = torch.exp(1.0 / bw * prob_mat)
            A = ww * K
            loss += torch.sum(A)

        if self.use_sqrt:
            return torch.sqrt(loss) if loss > 0 else 0 * loss
        else:
            return loss if loss > 0 else 0 * loss

    def forward(self, predict, target, gt, ignore_mask):
        visible_index = torch.where(ignore_mask.view(-1) == True)[0] # visible indexes ranging from 0 to N*H*W-1
        assert (visible_index.shape[0] != 0)
        predict_flat = predict.permute(0,2,3,1).contiguous().view(-1, predict.shape[1])[visible_index] # (N*H*W, C)[visible]
        target_flat = target.permute(0,2,3,1).contiguous().view(-1, target.shape[1])[visible_index] # (N*H*W, C)[visible]
        class_flat = gt.view(-1)[visible_index] # (N*H*W)[visible]

        loss, begin, class_count, temp = 0, 0, 0, class_flat[0]
        for i, class_i in enumerate(class_flat):
            assert (class_i != self.ignore)
            if class_i == temp:
                continue
            this_class_loss = self.cal_loss(predict_flat[begin:i], target_flat[begin:i])
            loss += this_class_loss
            class_count += 1
            begin, temp = i, class_i
        last_class_loss = self.cal_loss(predict_flat[begin:], target_flat[begin:])
        loss += last_class_loss
        class_count += 1
        loss /= class_count
        return loss


class ClsLoss(nn.Module):
    def __init__(self, opt):
        super(ClsLoss, self).__init__()
        self.criterionCLS = nn.CrossEntropyLoss(ignore_index=opt['ignore_index'])

    def forward(self, net, feat, gt, ignore_mask):
        cls_score = net(feat, 'cls')
        loss_cls = self.criterionCLS(cls_score, gt)

        with torch.no_grad():
            cls_score_ = cls_score.permute(0,2,3,1).contiguous()
            _, indices = torch.sort(cls_score_, descending=True)
            indices_ = indices.long()
            indices_.detach_()

            pred_class = torch.argmax(cls_score, dim=1).long()
            pred_class.detach_()

            correct_count = (pred_class==gt).long().sum()
            totol_count = ignore_mask.long().sum()
            assert (correct_count <= totol_count)
            acc_cls = correct_count.float() / totol_count

        return loss_cls, acc_cls, pred_class, indices_


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss(reduce=False)
        else:
            self.loss = nn.BCELoss(reduce=False)

    def get_target_tensor(self, pred, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != pred.numel()))
            if create_label:
                with torch.no_grad():
                    self.real_label_var = self.Tensor(pred.size()).fill_(self.real_label).cuda()
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != pred.numel()))
            if create_label:
                with torch.no_grad():
                    self.fake_label_var = self.Tensor(pred.size()).fill_(self.fake_label).cuda()
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, pred, target_is_real, ignore_mask):
        target_tensor = self.get_target_tensor(pred, target_is_real)
        mask = ignore_mask.float().unsqueeze(1).repeat(1, pred.shape[1], 1, 1)
        return (self.loss(pred, target_tensor) * mask).sum() / mask.sum()


class DiscLoss(nn.Module):
    def __init__(self):
        super(DiscLoss, self).__init__()
        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, net, predict, ignore_mask):
        pred_fake = net(predict, mode='gan')
        loss_G_GAN = self.criterionGAN(pred_fake, 1, ignore_mask)
        return loss_G_GAN

    def forward(self, net, predict, target, ignore_mask):
        pred_fake = net(predict, mode='gan')
        loss_D_fake = self.criterionGAN(pred_fake, 0, ignore_mask)

        pred_real = net(target, mode='gan')
        loss_D_real = self.criterionGAN(pred_real, 1, ignore_mask)

        loss_D = loss_D_fake + loss_D_real
        return loss_D, loss_D_real, loss_D_fake, None

class DiscLossLS(nn.Module):
    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, predict, ignore_mask):
        pred_fake = net(predict, mode='gan')
        loss_G_GAN = self.criterionGAN(pred_fake, 1, ignore_mask)
        return loss_G_GAN

    def forward(self, net, predict, target, ignore_mask):
        pred_fake = net(predict, mode='gan')
        loss_D_fake = self.criterionGAN(pred_fake, 0, ignore_mask)

        pred_real = net(target, mode='gan')
        loss_D_real = self.criterionGAN(pred_real, 1, ignore_mask)

        loss_D = loss_D_fake + loss_D_real
        return loss_D, loss_D_real, loss_D_fake, None

class DiscLossWGAN(nn.Module):
    def __init__(self, opt):
        super(DiscLossWGAN, self).__init__()
        self.LAMBDA = opt['lambda_D_gp']

    def get_g_loss(self, net, predict, ignore_mask):
        mask = ignore_mask.float().unsqueeze(1).repeat(1, predict.shape[1], 1, 1)

        pred_fake = net(predict, mode='gan')
        loss_G_GAN = -(pred_fake * mask).sum() / mask.sum()
        return loss_G_GAN

    def forward(self, net, predict, target, ignore_mask):
        mask = ignore_mask.float().unsqueeze(1).repeat(1, predict.shape[1], 1, 1)

        pred_fake = net(predict, mode='gan')
        loss_D_fake = (pred_fake * mask).sum() / mask.sum()

        pred_real = net(target, mode='gan')
        loss_D_real = -(pred_real * mask).sum() / mask.sum()

        loss_D = loss_D_fake + loss_D_real
        return loss_D, loss_D_real, loss_D_fake, None

class DiscLossWGANGP(nn.Module):
    def __init__(self, opt):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = opt['lambda_D_gp']

    def get_g_loss(self, net, predict, ignore_mask):
        mask = ignore_mask.float().unsqueeze(1).repeat(1, predict.shape[1], 1, 1)

        pred_fake = net(predict, mode='gan')
        loss_G_GAN = -(pred_fake * mask).sum() / mask.sum()
        return loss_G_GAN

    def calc_gradient_penalty(self, net, target, predict, mask):
        Tensor = torch.cuda.FloatTensor
        BATCH_SIZE = target.shape[0]

        alpha = torch.rand(BATCH_SIZE, 1, 1, 1).cuda().expand(target.shape)
        interpolates = autograd.Variable(alpha * target + (1 - alpha) * predict, requires_grad=True)
        disc_interpolates = net(interpolates, mode='gan')
        grad_outputs = autograd.Variable(Tensor(disc_interpolates.shape).fill_(1.0), requires_grad=False)

        gradients = autograd.grad(
            outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=grad_outputs,
            retain_graph=True, 
            create_graph=True, 
            only_inputs=True, 
            allow_unused=False
        )[0]

        gradients_flat = gradients.view(BATCH_SIZE, -1)
        mask_flat = mask.view(BATCH_SIZE, -1)
        gradients_visible = gradients_flat * mask_flat

        gradient_penalty = ((gradients_visible.norm(p=2, dim=1) - 1) ** 2).mean() * self.LAMBDA

        return gradient_penalty

    def forward(self, net, predict, target, ignore_mask):
        mask = ignore_mask.float().unsqueeze(1).repeat(1, predict.shape[1], 1, 1)

        pred_fake = net(predict, mode='gan')
        loss_D_fake = (pred_fake * mask).sum() / mask.sum()

        pred_real = net(target, mode='gan')
        loss_D_real = -(pred_real * mask).sum() / mask.sum()

        gradient_penalty = self.calc_gradient_penalty(net, target, predict, mask)

        loss_D = loss_D_fake + loss_D_real + gradient_penalty
        return loss_D, loss_D_real, loss_D_fake, gradient_penalty


def init_loss(opt):
    if opt['content_loss'] == 'PerceptualLoss':
        content_loss = PerceptualLoss(nn.MSELoss(reduce=False))
    elif opt['content_loss'] == 'ContentLoss':
        content_loss = ContentLoss(nn.L1Loss(reduce=False))
    elif opt['content_loss'] == 'ContentLossMSE':
        content_loss = ContentLoss(nn.MSELoss(reduce=False))
    elif opt['content_loss'] == 'MMDLoss':
        content_loss = MMDLoss(opt, use_sqrt=True)
    else:
        raise ValueError("Content loss [%s] not recognized." % opt['content_loss'])

    if opt['gan_type'] == 'wgan-gp':
        disc_loss = DiscLossWGANGP(opt) 
    elif opt['gan_type'] == 'wgan':
        disc_loss = DiscLossWGAN(opt)
    elif opt['gan_type'] == 'lsgan':
        disc_loss = DiscLossLS()
    elif opt['gan_type'] == 'gan':
        disc_loss = DiscLoss()
    else:
        raise ValueError("GAN type [%s] not recognized." % opt['gan_type'])

    cls_loss = ClsLoss(opt)

    return disc_loss, content_loss, cls_loss