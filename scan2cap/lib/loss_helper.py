# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "utils")) # HACK add the utils folder
sys.path.append(os.path.join(os.getcwd(), "utils/pycocoevalcap"))

from utils.pycocoevalcap.bleu.bleu import Bleu
from utils.meteor import *
from utils.pycocoevalcap.rouge.rouge import Rouge
from utils.pycocoevalcap.cider.cider import Cider

def pointnet_pretrain_loss(data_dict):
    target = data_dict["ref_nyu40_label"]
    scores = data_dict["ref_obj_cls_scores"]
    loss = F.cross_entropy(scores, target-1, weight=data_dict["class_weights"][0])
    data_dict["loss"] = loss

    _, preds = torch.max(scores, dim=1)
    preds += 1
    acc = torch.mean((preds == target).to(dtype=torch.float32))
    data_dict["ref_acc"] = acc
    data_dict["scores"] = preds

    return loss, data_dict

def caption_loss(data_dict, vocabulary):

    targets = data_dict["lang_indices"]
    
    scores = data_dict["caption_predictions"]
      
    loss = F.cross_entropy(scores, targets, ignore_index=-1)
    data_dict["loss"] = loss

    references = {}
    hypotheses = {}

    hypo = torch.argmax(scores, dim=1)

    for i in range(targets.size(0)):
        #stringify
        target_strings = [" ".join([vocabulary[index] for index in targets[i] if index > 0])]

        num_oth_ref = data_dict["other_lang_indices"].size(1)
        oth_ref = [
            ' '.join([vocabulary[index] for index in data_dict["other_lang_indices"][i,t,:] if index > 0])
            for t in range(num_oth_ref)]
        references["{}".format(i)] = oth_ref

        hypo_strings = [vocabulary[index] for index in hypo[i] if index > 0]
        hypotheses["{}".format(i)] = [' '.join(hypo_strings)]

    #print("Ref:", references["0"][0])
    #print("Hyp:", hypotheses["0"][0])

    bleu4, _ = Bleu(n=4).compute_score(references, hypotheses)
    meteor = compute_meteor(references, hypotheses)
    rouge, _ = Rouge().compute_score(references, hypotheses)
    cider, _ = Cider().compute_score(references, hypotheses)


    #print(bleu4, meteor, rouge, cider)

    #meteor = 0
    data_dict["bleu4"] = bleu4[3]
    data_dict["rouge"]= rouge
    data_dict["meteor"] = meteor
    data_dict["cider"] = cider
    if "alphas" in data_dict:
        att = data_dict["alphas"].detach().cpu().numpy()
        att_max = np.max(att)
        att_var = np.var(att, axis=-1)
        att_var = np.mean(att_var)
    else:
        att_max = 0
        att_var = 0
    data_dict["attention_max"] = att_max
    data_dict["attention_var"] = att_var

    caption_length_gen = 0
    caption_length_gt = 0
    num_sent_val = []
    num_sent_train = []

    #calculate dataset metrics 
    if "caption_indices" in data_dict:
        for ci in data_dict["caption_indices"].detach().cpu().numpy():
            num_sent_val.append(sum([(x==17 or x==15) for x in ci]))
        data_dict["mean_sentence_length"] = np.mean(np.asarray(num_sent_val))
        batch_caption_lengths = (data_dict["caption_indices"] > 0).to(dtype=torch.int32).sum(dim=1)    
        data_dict["batch_caption_lenghts"] = batch_caption_lengths
        caption_length_gen += batch_caption_lengths.sum()
        caption_length_gt += (data_dict["lang_len"] - 1).sum()
        data_dict["caption_ratio"] = (caption_length_gen / caption_length_gt).detach().cpu().numpy()
    else:
        for li in data_dict["lang_indices"].detach().cpu().numpy():
            num_sent_train.append(sum([(x==17 or x==15) for x in li]))
        data_dict["mean_sentence_length"] = np.mean(np.asarray(num_sent_train))
        data_dict["caption_ratio"] = np.array([1])
    return loss, data_dict


def attention_regularization(data_dict, alpha_c=1.):
    data_dict["loss"] += alpha_c * ((1. - data_dict["alphas"].sum(dim=1)) ** 2).mean()
    return data_dict
