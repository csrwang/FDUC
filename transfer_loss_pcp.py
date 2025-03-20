import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import gc

### 渐进式原型对齐版本


def CalculateMean(features, labels, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()


def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()




def MO(mean_source_up1, cv_source_up1, features_target1, logits_gnn_target, class_num):

    ce_criterion = nn.CrossEntropyLoss()

    N = features_target1.size(0)
    C = class_num
    A = features_target1.size(1)


    ### 归一化1
    # features_target = (features_target1 - features_target1.mean(dim=1, keepdim=True)) / features_target1.std(dim=1, keepdim=True)
    # mean_source_up = (mean_source_up1 - mean_source_up1.mean(dim=1, keepdim=True)) / mean_source_up1.std(dim=1, keepdim=True)
    # cv_source_up = (cv_source_up1 - cv_source_up1.mean(dim=[1, 2], keepdim=True)) / cv_source_up1.std(dim=[1, 2], keepdim=True)

    ### 归一化2
    # features_target = features_target1 / features_target1.norm(dim=1, keepdim=True)
    # mean_source_up= mean_source_up1 / mean_source_up1.norm(dim=1, keepdim=True)
    # cv_source_up = cv_source_up1.view(C, -1).div(cv_source_up1.view(C, -1).norm(dim=1, keepdim=True)).view(C, A, A)

    # 计算范数，并替换0值为1，以避免除以0的情况
    norm_features_target1 = features_target1.norm(dim=1, keepdim=True)
    norm_features_target1[norm_features_target1 == 0] = 1
    features_target = features_target1 / norm_features_target1

    norm_mean_source_up1 = mean_source_up1.norm(dim=1, keepdim=True)
    norm_mean_source_up1[norm_mean_source_up1 == 0] = 1
    mean_source_up = mean_source_up1 / norm_mean_source_up1

    norm_cv_source_up1 = cv_source_up1.view(C, -1).norm(dim=1, keepdim=True)
    norm_cv_source_up1[norm_cv_source_up1 == 0] = 1
    cv_source_up = cv_source_up1.view(C, -1) / norm_cv_source_up1
    cv_source_up = cv_source_up.view(C, A, A)



    # ### 不归一化
    # features_target = features_target1
    # mean_source_up = mean_source_up1
    # cv_source_up = cv_source_up1


    _, predict_gnn_target = torch.max(logits_gnn_target, 1)

    # sourceMean_NxA = mean_source_up[predict_gnn_target]   ##32x256
    # sourceMean_NxCxA = sourceMean_NxA.expand(N, C, A)    ##32x31x256

    ## calculate g_mu
    sourceMean_NxCxA = mean_source_up.expand(N, C, A)    ##32x31x256  矩阵是每个类别的mean，而不是根据logits_gnn_target标签为每个样本找到的mean。 mean_source_up：31x256。
    sourceMean_NxAxC = sourceMean_NxCxA.permute(0, 2, 1)   ##32x256x31

    features_target_Nx1xA = features_target.unsqueeze(1)    ##32x1x256， features_target是32x256

    g_mu = torch.bmm(features_target_Nx1xA, sourceMean_NxAxC)  ##32x31
    g_mu = g_mu.squeeze(1)

    ## calculate g_sigma2_g
    sourceCV_NxCxAxA = cv_source_up.expand(N, C, A, A)   ## 32x31x256x256
    features_target_NxCx1xA = features_target.unsqueeze(1).unsqueeze(2)  ## 32x1x1x256
    features_target_NxCx1xA = features_target_NxCx1xA.expand(-1, C, -1, -1)   ## 32x31x1x256
    features_target_NxCxAx1 = features_target_NxCx1xA.permute(0, 1, 3, 2) ## 32x31x256x1

    result = torch.matmul(features_target_NxCx1xA, sourceCV_NxCxAxA)
    g_sigma2_g = torch.matmul(result, features_target_NxCxAx1)
    g_sigma2_g = g_sigma2_g.squeeze()

    # aug_result = g_mu + 0.5 * g_sigma2_g
    aug_result = g_mu


    loss = ce_criterion(aug_result, predict_gnn_target)

    return loss


