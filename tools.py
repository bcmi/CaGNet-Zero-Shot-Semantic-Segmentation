import os
import sys
import yaml
import random
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class MeaninglessError(BaseException):
    pass

class Const_Scheduler():
    def __init__(self, step_n='step1'):
        assert (step_n in ['step1', 'step2', 'self_training'])
        self.step_n = step_n
        pass

    def now(self):
        return self.step_n

    def step(self):
        pass

class Step_Scheduler():
    def __init__(self, interval_step1, interval_step2, first='step2'):
        assert (first in ['step1', 'step2'])
        assert (interval_step1 > 0 and interval_step2 > 0)
        self.interval_step1 = int(interval_step1)
        self.interval_step2 = int(interval_step2)
        self.first = first
        self.now_step = 0

    def now(self):
        assert (self.now_step in range(self.interval_step1 + self.interval_step2))

        if self.first == 'step2':
            if self.now_step < self.interval_step2:
                return 'step2'
            else:
                return 'step1'
        else:
            if self.now_step < self.interval_step1:
                return 'step1'
            else:
                return 'step2'

    def step(self):
        self.now_step += 1
        if self.now_step == self.interval_step1 + self.interval_step2:
            self.now_step = 0


class logWritter():
    def __init__(self, log_file):
        self.logs =  log_file
        if not os.path.exists(log_file):
            os.mknod(log_file)
            
    def write(self, strs):
        assert (type(strs) == str)
        with open(self.logs, 'a') as f:
            f.write(strs + '\n')

class RandomImageSampler(torch.utils.data.Sampler):
    """
    Samples classes randomly, then returns images corresponding to those classes.
    """

    def __init__(self, seenset, novelset):
        self.data_index = []
        for v in seenset:
            self.data_index.append([v, 0])
        for v,i in novelset:
            self.data_index.append([v, i+1])

    def __iter__(self):
        return iter([ self.data_index[i] for i in np.random.permutation(len(self.data_index))])

    def __len__(self):
        return len(self.data_index)

def construct_gt_st(resized_gt_st, sorted_indices, config):
    indices_select = sorted_indices[:,:,:,:config['top_p']] # retain category indices with top_p prediction scores
    indices_select_pos = torch.full(indices_select.shape, config['ignore_index']).long()
    indices_select_neg = torch.full(indices_select.shape, -config['ignore_index']).long()
    indices_repeat = torch.LongTensor(range(config['top_p'])).repeat(indices_select.shape[0],indices_select.shape[1],indices_select.shape[2],1)
    p0 = torch.where(indices_select >= config['dis']['out_dim_cls']-config['num_unseen']-1, indices_select, indices_select_pos).long()
    p1 = torch.where(indices_select < config['dis']['out_dim_cls']-1, indices_select, indices_select_neg).long()
    p2 = torch.where(p0 == p1, indices_select, indices_select_pos).long()
    p3 = torch.where(p0 == p1, indices_repeat, indices_select_pos).long()
    p4 = torch.argmin(p3, dim=3).long()
    accumulated = config['top_p'] * torch.LongTensor(range(p2.shape[0]*p2.shape[1]*p2.shape[2]))
    p5 = p4.view(-1) + accumulated
    p6 = p2.view(-1)[p5].view(resized_gt_st.shape)
    gt_new = torch.where(resized_gt_st == config['ignore_index'], p6, resized_gt_st).long()
    return gt_new

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.cpu().numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_embedding(cfg):
    dataset_path = os.path.join(cfg['datadir'], cfg['dataset'])
    if  cfg['embedding'] == 'word2vec':
        class_emb = pickle.load(open(dataset_path+'/word_vectors/word2vec.pkl', "rb"))
    elif cfg['embedding'] == 'fasttext':
        class_emb = pickle.load(open(dataset_path+'/word_vectors/fasttext.pkl', "rb"))
    elif cfg['embedding'] == 'fastnvec':
        class_emb = np.concatenate([pickle.load(open(dataset_path+'/word_vectors/fasttext.pkl', "rb")), pickle.load(open(dataset_path+'/word_vectors/word2vec.pkl', "rb"))], axis = 1)
    else:
        print("invalid embedding: {0}".format(cfg['embedding']))
        sys.exit() 

    if not cfg['emb_without_normal']:
        class_emb = F.normalize(torch.tensor(class_emb, dtype = torch.float32), p=2, dim=1)
        print("Class embedding map normalized!")
    else:
        class_emb = torch.tensor(class_emb, dtype = torch.float32)
    return class_emb

def get_split(cfg):
    dataset_path = os.path.join(cfg['datadir'], cfg['dataset'])
    train = np.load(dataset_path + '/split/train_list.npy')
    val = np.load(dataset_path + '/split/test_list.npy')

    seen_classes = np.load(dataset_path + '/split/seen_cls.npy').astype(np.int32)
    novel_classes = np.load(dataset_path + '/split/novel_cls.npy').astype(np.int32)
    seen_novel_classes = np.concatenate((seen_classes, novel_classes), axis=0)
    all_labels  = np.genfromtxt(dataset_path + '/labels_2.txt', delimiter='\t', usecols=1, dtype='str')

    visible_classes = seen_classes
    visible_classes_test = seen_novel_classes

    novelset, seenset = [], range(train.shape[0])
    sampler = RandomImageSampler(seenset, novelset)

    cls_map = np.array([cfg['ignore_index']]*(cfg['ignore_index']+1)).astype(np.int32)
    for i, n in enumerate(list(seen_classes)):
        cls_map[n] = i
    cls_map_test = np.array([cfg['ignore_index']]*(cfg['ignore_index']+1)).astype(np.int32)
    for i, n in enumerate(list(seen_novel_classes)):
        cls_map_test[n] = i

    visibility_mask = {}
    visibility_mask[0] = cls_map.copy()
    for i, n in enumerate(list(novel_classes)):
        visibility_mask[i+1] = cls_map.copy()
        visibility_mask[i+1][n] = seen_classes.shape[0] + i

    return seen_classes, novel_classes, all_labels, visible_classes, visible_classes_test, train, val, sampler, visibility_mask, cls_map, cls_map_test

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
    }, cls_iu

def scores_gzsl(label_trues, label_preds, n_class, seen_cls, unseen_cls):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(hist).sum() / hist.sum()
        seen_acc = np.diag(hist)[seen_cls].sum() / hist[seen_cls].sum()
        unseen_acc = np.diag(hist)[unseen_cls].sum() / hist[unseen_cls].sum()
        h_acc = 2./(1./seen_acc + 1./unseen_acc)
        if np.isnan(h_acc):
            h_acc = 0
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        seen_acc_cls = np.diag(hist)[seen_cls] / hist.sum(axis=1)[seen_cls]
        unseen_acc_cls = np.diag(hist)[unseen_cls] / hist.sum(axis=1)[unseen_cls]
        acc_cls = np.nanmean(acc_cls)
        seen_acc_cls = np.nanmean(seen_acc_cls)
        unseen_acc_cls = np.nanmean(unseen_acc_cls)
        h_acc_cls = 2./(1./seen_acc_cls + 1./unseen_acc_cls)
        if np.isnan(h_acc_cls):
            h_acc_cls = 0
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        seen_mean_iu = np.nanmean(iu[seen_cls])
        unseen_mean_iu = np.nanmean(iu[unseen_cls])
        h_mean_iu = 2./(1./seen_mean_iu + 1./unseen_mean_iu)
        if np.isnan(h_mean_iu):
            h_mean_iu = 0
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq * iu)
        fwavacc[np.isnan(fwavacc)] = 0
        seen_fwavacc = fwavacc[seen_cls].sum()
        unseen_fwavacc = fwavacc[unseen_cls].sum()
        h_fwavacc = 2./(1./seen_fwavacc + 1./unseen_fwavacc)
        if np.isnan(h_fwavacc):
            h_fwavacc = 0
        fwavacc = fwavacc.sum()
        cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Overall Acc Seen": seen_acc,
        "Overall Acc Unseen": unseen_acc,
        "Overall Acc Harmonic": h_acc,
        "Mean Acc": acc_cls,
        "Mean Acc Seen": seen_acc_cls,
        "Mean Acc Unseen": unseen_acc_cls,
        "Mean Acc Harmonic": h_acc_cls,
        "FreqW Acc": fwavacc,
        "FreqW Acc Seen": seen_fwavacc,
        "FreqW Acc Unseen": unseen_fwavacc,
        "FreqW Acc Harmonic": h_fwavacc,
        "Mean IoU": mean_iu,
        "Mean IoU Seen": seen_mean_iu,
        "Mean IoU Unseen": unseen_mean_iu,
        "Mean IoU Harmonic": h_mean_iu,
    }, cls_iu