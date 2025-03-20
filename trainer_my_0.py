import torch
import torch.nn as nn
import networks
import transfer_loss
from preprocess import ImageList, ConcatDataset
from torch.utils.data import DataLoader
import utils
import main_dcgct_my_0  #### add
from main_dcgct_my_0 import DEVICE
# from transfer_loss_mo import *
from transfer_loss_pcp import *
import scipy.io as scio




def evaluate(i, config, base_network, classifier_gnn, target_test_dset_dict, list_acc):
    base_network.eval()
    classifier_gnn.eval()
    mlp_accuracy_list, gnn_accuracy_list = [], []
    for dset_name, test_loader in target_test_dset_dict.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        mlp_accuracy, gnn_accuracy = test_res['mlp_accuracy'], test_res['gnn_accuracy']
        mlp_accuracy_list.append(mlp_accuracy)
        gnn_accuracy_list.append(gnn_accuracy)
        # print out test accuracy for domain
        log_str = 'Dataset:%s\tTest Accuracy mlp %.4f\tTest Accuracy gnn %.4f' \
                  % (dset_name, mlp_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)

    # print out domains averaged accuracy
    mlp_accuracy_avg = sum(mlp_accuracy_list) / len(mlp_accuracy_list)
    gnn_accuracy_avg = sum(gnn_accuracy_list) / len(gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy MLP Classifier: %.4f, Avg Accuracy GNN classifier: %.4f' \
              % (i, mlp_accuracy_avg * 100., gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    if mlp_accuracy_avg >= gnn_accuracy_avg:
        temp_acc = mlp_accuracy_avg
    else:
        temp_acc = gnn_accuracy_avg
    if temp_acc >= main_dcgct_my_0.max_accuracy:
        main_dcgct_my_0.max_accuracy = temp_acc
    log_str = 'Max Accuracy: %.4f' \
              % (main_dcgct_my_0.max_accuracy * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    list_acc.append(gnn_accuracy_avg * 100.)
    id = np.argmax(np.array(list_acc))
    max_acc = list_acc[id]
    log_str = 'max_acc: %.4f' % (max_acc)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)


    base_network.train()
    classifier_gnn.train()


def eval_domain(config, test_loader, base_network, classifier_gnn):
    logits_mlp_all, logits_gnn_all, confidences_gnn_all, labels_all = [], [], [], []
    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = next(iter_test)
            inputs = data['img'].to(DEVICE)
            # forward pass
            feature, logits_mlp = base_network(inputs)
            # check if number of samples is greater than 1
            if len(inputs) == 1:
                # gnn cannot handle only one sample ... use MLP instead
                # this can be encountered if len_dataset % test_batch == 1
                logits_gnn = logits_mlp
            else:
                logits_gnn, _ = classifier_gnn(feature)
            logits_mlp_all.append(logits_mlp.cpu())
            logits_gnn_all.append(logits_gnn.cpu())
            confidences_gnn_all.append(nn.Softmax(dim=1)(logits_gnn_all[-1]).max(1)[0])
            labels_all.append(data['target'])

    # concatenate data
    logits_mlp = torch.cat(logits_mlp_all, dim=0)
    logits_gnn = torch.cat(logits_gnn_all, dim=0)
    confidences_gnn = torch.cat(confidences_gnn_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    # predict class labels
    _, predict_mlp = torch.max(logits_mlp, 1)
    _, predict_gnn = torch.max(logits_gnn, 1)


    mlp_accuracy = torch.sum(predict_mlp == labels).item() / len(labels)
    gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)

    # compute mask for high confident samples
    sample_masks_bool = (confidences_gnn > config['threshold'])

    sample_masks_idx = torch.nonzero(sample_masks_bool, as_tuple=True)[0].numpy()
    # compute accuracy of pseudo labels
    total_pseudo_labels = len(sample_masks_idx)
    if len(sample_masks_idx) > 0:
        correct_pseudo_labels = torch.sum(predict_gnn[sample_masks_bool] == labels[sample_masks_bool]).item()
        pseudo_label_acc = correct_pseudo_labels / total_pseudo_labels
    else:
        correct_pseudo_labels = -1.
        pseudo_label_acc = -1.
    out = {
        'mlp_accuracy': mlp_accuracy,
        'gnn_accuracy': gnn_accuracy,
        'confidences_gnn': confidences_gnn,
        'pred_cls': predict_gnn.numpy(),
        'sample_masks': sample_masks_idx,
        'sample_masks_cgct': sample_masks_bool.float(),
        'pseudo_label_acc': pseudo_label_acc,
        'correct_pseudo_labels': correct_pseudo_labels,
        'total_pseudo_labels': total_pseudo_labels,
    }
    return out


def evaluate_new(i, config, base_network, classifier_gnn, target_test_dset_dict, mean_domain_up, list_acc):
    base_network.eval()
    classifier_gnn.eval()
    mlp_accuracy_list, gnn_accuracy_list = [], []
    for dset_name, test_loader in target_test_dset_dict.items():
        test_res = eval_domain_new(config, test_loader, base_network, classifier_gnn, mean_domain_up)
        mlp_accuracy, gnn_accuracy = test_res['mlp_accuracy'], test_res['gnn_accuracy']
        mlp_accuracy_list.append(mlp_accuracy)
        gnn_accuracy_list.append(gnn_accuracy)
        # print out test accuracy for domain
        log_str = 'Dataset:%s\tTest Accuracy mlp %.4f\tTest Accuracy gnn %.4f' \
                  % (dset_name, mlp_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)

    # print out domains averaged accuracy
    mlp_accuracy_avg = sum(mlp_accuracy_list) / len(mlp_accuracy_list)
    gnn_accuracy_avg = sum(gnn_accuracy_list) / len(gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy MLP Classifier: %.4f, Avg Accuracy GNN classifier: %.4f' \
              % (i, mlp_accuracy_avg * 100., gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    if mlp_accuracy_avg >= gnn_accuracy_avg:
        temp_acc = mlp_accuracy_avg
    else:
        temp_acc = gnn_accuracy_avg
    if temp_acc >= main_dcgct_my_0.max_accuracy:
        main_dcgct_my_0.max_accuracy = temp_acc
    log_str = 'Max Accuracy: %.4f' \
              % (main_dcgct_my_0.max_accuracy * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    list_acc.append(gnn_accuracy_avg * 100.)
    id = np.argmax(np.array(list_acc))
    max_acc = list_acc[id]
    log_str = 'max_acc: %.4f' % (max_acc)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    base_network.train()
    classifier_gnn.train()


def eval_domain_new(config, test_loader, base_network, classifier_gnn, mean_domain_up):
    y_causal_log_mean_softmax_mlp_all, y_causal_log_mean_softmax_gnn_all, confidences_gnn_all, labels_all = [], [], [], []
    n_domains = config['encoder_new']['params']['domain_num']

    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = next(iter_test)
            inputs = data['img'].to(DEVICE)
            domain_labels = data['flag'].to(DEVICE)

            # forward pass
            x_original, mu_1, log_var_1, mu_2, log_var_2, x1_new, x2_new, y_x1_1, y_x1_2, y_x2_1, y_x2_2, x_reconstructed, x_causal_feature_all, y_causal_log_mean_softmax \
                                                                            = base_network(inputs, Ci=mean_domain_up, step=3, domainlabel=domain_labels)

            y_causal_log_mean_softmax_mlp = y_causal_log_mean_softmax
            # check if number of samples is greater than 1
            if len(inputs) == 1:
                # gnn cannot handle only one sample ... use MLP instead
                # this can be encountered if len_dataset % test_batch == 1
                y_causal_log_mean_softmax_gnn = y_causal_log_mean_softmax_mlp
            else:
                # logits_gnn, _ = classifier_gnn(feature)

                ## tiao
                y_causal_log_mean_softmax_gnn = causal_gcn_test(classifier_gnn, x_causal_feature_all, n_domains)
                # logits_gnn = causal_gcn_test_c1(classifier_gnn, x_reconstructed_causal, n_domains)

            y_causal_log_mean_softmax_mlp_all.append(y_causal_log_mean_softmax_mlp.cpu())
            y_causal_log_mean_softmax_gnn_all.append(y_causal_log_mean_softmax_gnn.cpu())
            labels_all.append(data['target'])

    # concatenate data
    y_causal_log_mean_softmax_mlp_all = torch.cat(y_causal_log_mean_softmax_mlp_all, dim=0)
    y_causal_log_mean_softmax_gnn_all = torch.cat(y_causal_log_mean_softmax_gnn_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    # predict class labels
    _, predict_mlp = torch.max(torch.exp(y_causal_log_mean_softmax_mlp_all), 1)
    _, predict_gnn = torch.max(torch.exp(y_causal_log_mean_softmax_gnn_all), 1)

    mlp_accuracy = torch.sum(predict_mlp == labels).item() / len(labels)
    gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)

    out = {
        'mlp_accuracy': mlp_accuracy,
        'gnn_accuracy': gnn_accuracy
    }

    return out


def causal_gcn_test(classifier_gnn, x_causal_feature_all, n_domains):

    feature_dim = x_causal_feature_all.size(1)
    x_reconstructed_causal_reshaped = x_causal_feature_all.view(n_domains, -1, feature_dim)

    ## 循环
    total_logits_list = []

    for domain in range(n_domains):
        # 提取每个domain对应的 classxfeature_dim 维度的矩阵
        x_reconstructed_causal_domain = x_reconstructed_causal_reshaped[domain]

        logits_gnn, _ = classifier_gnn(x_reconstructed_causal_domain)

        # 累加每次循环的结果
        total_logits_list.append(logits_gnn)

    ### log_mean_softmax
    total_logits_tensor = torch.stack(total_logits_list)
    y_causal_grouped_log_softmax = F.log_softmax(total_logits_tensor, dim=2)

    log_domain = torch.log(torch.tensor(n_domains))
    y_causal_log_mean_softmax = torch.logsumexp(y_causal_grouped_log_softmax, dim=0) - log_domain

    return y_causal_log_mean_softmax


def causal_gcn_test_pm(classifier_gnn, x_reconstructed_causal, n_domains):

    feature_dim = x_reconstructed_causal.size(1)
    x_reconstructed_causal_reshaped = x_reconstructed_causal.view(n_domains, -1, feature_dim)
    ## 循环
    total_logits_gnn = 0

    for domain in range(n_domains):
        # 提取每个domain对应的 classxfeature_dim 维度的矩阵
        x_reconstructed_causal_domain = x_reconstructed_causal_reshaped[domain]

        logits_gnn, _ = classifier_gnn(x_reconstructed_causal_domain)

        # 累加每次循环的结果
        total_logits_gnn += logits_gnn

    average_logits_gnn = total_logits_gnn / n_domains

    return average_logits_gnn


def causal_gcn_test_c1(classifier_gnn, x_reconstructed_causal, n_domains):

    logits_gnn, _ = classifier_gnn(x_reconstructed_causal)

    return logits_gnn


def select_closest_domain(config, base_network, classifier_gnn, temp_test_loaders):
    """
    This function selects the closest domain (Stage 2 in Algorithm 2 of Supp Mat) where adaptation need to be performed.
    In the code we compute the mean of the max probability of the target samples from a domain, which can be
    considered as inversely proportional to the mean of the entropy.

    Higher the max probability == lower is the entropy == higher the inheritability/similarity
    """
    base_network.eval()
    classifier_gnn.eval()
    max_inherit_val = 0.
    for dset_name, test_loader in temp_test_loaders.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        domain_inheritability = test_res['confidences_gnn'].mean().item()

        if domain_inheritability > max_inherit_val:
            max_inherit_val = domain_inheritability
            max_inherit_domain_name = dset_name

    print('Most similar target domain: %s' % (max_inherit_domain_name))
    log_str = 'Most similar target domain: %s' % (max_inherit_domain_name)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    return max_inherit_domain_name


def train_source(config, base_network, classifier_gnn, dset_loaders):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() + \
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    ## addxu memory bank
    memory_source_features = torch.zeros(len(dset_loaders["source"].dataset), 256).cuda()
    memory_source_labels = torch.zeros(len(dset_loaders["source"].dataset)).long().cuda()
    memory_domain_labels = torch.zeros(len(dset_loaders["source"].dataset)).long().cuda()

    flag = False
    for _, sample in enumerate(dset_loaders["source"]):
        images = sample['img'].cuda()  # 从字典中提取图片并传输到GPU
        label = sample['target'].cuda()  # 从字典中提取标签并传输到GPU
        index = sample['idx']  # 从字典中提取索引
        domain_label = sample['flag'].cuda()  # 从字典中提取标签并传输到GPU

        # 检查是否只有一张图片
        if images.size(0) == 1:
            temp_iter = iter(dset_loaders["source"])
            next_sample = next(temp_iter)
            images_a = next_sample['img'].cuda()  # 从字典中提取图片并传输到GPU
            images = torch.cat((images, images_a), dim=0)
            flag = True
            del temp_iter
            del _
        with torch.no_grad():
            features_temp, _ = base_network(images)
            del _
            if flag:
                memory_source_features[index] = features_temp[0].unsqueeze(0)
                memory_source_labels[index] = label
                memory_domain_labels[index] = domain_label
                flag = False
            else:
                memory_source_features[index] = features_temp
                memory_source_labels[index] = label
                memory_domain_labels[index] = domain_label

            del features_temp

    print("memory module initialization has finished!")


    # start train loop
    base_network.train()
    classifier_gnn.train()
    len_train_source = len(dset_loaders["source"])

    #——————————————————
    list_acc = []
    # —————————————————

    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        batch_source = next(iter_source)
        inputs_source, labels_source = batch_source['img'].to(DEVICE), batch_source['target'].to(DEVICE)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp = base_network(inputs_source)
        mlp_loss = ce_criterion(logits_mlp, labels_source)

        ## addxu calculate mean CV
        class_num = config['encoder_new']['params']['class_num']
        batch_size = config['data']['source']['batch_size']
        cat_number = features_source.shape[0]

        # 首先移除 memory_source_features 和 memory_source_labels 的前 batch_size 个元素
        memory_source_features = memory_source_features[cat_number:]
        memory_source_labels = memory_source_labels[cat_number:]
        # 然后在尾部添加新的 features_source 和 labels_source
        memory_source_features = torch.cat((memory_source_features, features_source), dim=0)
        memory_source_labels = torch.cat((memory_source_labels, labels_source), dim=0)


        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features_source)
        gnn_loss = ce_criterion(logits_gnn, labels_source)
        # compute edge loss
        edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # total loss and backpropagation
        loss = mlp_loss + config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f' % (i,
                                                                                        config['source_iters'],
                                                                                        mlp_loss.item(),
                                                                                        gnn_loss.item(),
                                                                                        edge_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'], list_acc)

    # addxu 计算均值，协方差  可去掉
    mean_source = CalculateMean(memory_source_features, memory_source_labels, class_num)
    cv_source = Calculate_CV(memory_source_features, memory_source_labels, mean_source, class_num)
    print("memory module updata has finished!-step1")


    return base_network, classifier_gnn, mean_source, cv_source, memory_source_features, memory_source_labels, memory_domain_labels


def train_source_step3(config, base_network, classifier_gnn, dset_loaders):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() +\
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    len_train_source = len(dset_loaders["source"])

    #——————————————————
    list_acc = []
    # —————————————————

    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        batch_source = next(iter_source)
        inputs_source, labels_source = batch_source['img'].to(DEVICE), batch_source['target'].to(DEVICE)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp = base_network(inputs_source)   ### features_source 32x256, logits_mlp 32x31
        mlp_loss = ce_criterion(logits_mlp, labels_source)

        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features_source)
        gnn_loss = ce_criterion(logits_gnn, labels_source)
        # compute edge loss
        edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # total loss and backpropagation
        loss = mlp_loss + config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f' % (i,
                  config['source_iters'], mlp_loss.item(), gnn_loss.item(), edge_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'], list_acc)


    return base_network, classifier_gnn


def kl_anneal_function(anneal_cap, step, total_annealing_step=10000):

    return min(anneal_cap, step / total_annealing_step)


def train_source_step3_new(config, base_network, classifier_gnn, dset_loaders, memory_source_features, memory_domain_labels):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()
    ce_criterion_pro = nn.NLLLoss()

    kl_loss = nn.KLDivLoss(reduction='mean')

    n_domains = config['encoder_new']['params']['domain_num']
    class_num = config['encoder']['params']['class_num']
    batch_size = config['data']['source']['batch_size']

    memory_source_features_up = memory_source_features
    memory_domain_labels_up = memory_domain_labels
    mean_domain_up = CalculateMean(memory_source_features_up, memory_domain_labels_up, n_domains)



    ## fanzhuan , jianshao
    num_samples_s1 = len(dset_loaders['source'].dataset)
    cat_s1 = memory_source_features_up.shape[0] - num_samples_s1
    memory_source_features_up = torch.flip(memory_source_features_up, [0])
    memory_domain_labels_up = torch.flip(memory_domain_labels_up, [0])
    memory_source_features_up = memory_source_features_up[cat_s1:]
    memory_domain_labels_up = memory_domain_labels_up[cat_s1:]



    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() +\
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    len_train_source = len(dset_loaders["source"])

    #——————————————————
    list_acc = []
    # —————————————————

    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        batch_source = next(iter_source)
        inputs_source, labels_source = batch_source['img'].to(DEVICE), batch_source['target'].to(DEVICE)
        domain_labels = batch_source['flag'].to(DEVICE)

        cat_number = inputs_source.shape[0]

        #### addxu   # *** causal-Method ***
        if config['causal_flag'].startswith('false'):

        # make forward pass for encoder and mlp head
            features_source, logits_mlp = base_network(inputs_source)
            mlp_loss = ce_criterion(logits_mlp, labels_source)

            # make forward pass for gnn head
            logits_gnn, edge_sim = classifier_gnn(features_source)
            gnn_loss = ce_criterion(logits_gnn, labels_source)
            # compute edge loss
            edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))
            edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

            # total loss and backpropagation
            loss = mlp_loss + config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
            loss.backward()
            optimizer.step()

            # printout train loss
            if i % 20 == 0 or i == config['source_iters'] - 1:
                log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f' % (i,
                      config['source_iters'], mlp_loss.item(), gnn_loss.item(), edge_loss.item())
                utils.write_logs(config, log_str)
            # evaluate network every test_interval
            if i % config['test_interval'] == config['test_interval'] - 1:
                evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'], list_acc)

        elif config['causal_flag'].startswith('true'):

            x_original, mu_1, log_var_1, mu_2, log_var_2, x1_new, x2_new, y_x1_c1, y_x1_c2, y_x2_c1, y_x2_c2, x_reconstructed, x_causal_feature_all, y_causal_log_mean_softmax   \
                                                 = base_network(inputs_source, Ci=mean_domain_up, step=3, domainlabel=domain_labels)  ### Ci

            memory_source_features_up = memory_source_features_up[cat_number:]
            memory_domain_labels_up = memory_domain_labels_up[cat_number:]

            # 然后在尾部添加新的 features_source 和 labels_source
            memory_source_features_up = torch.cat((memory_source_features_up, x2_new), dim=0)
            memory_domain_labels_up = torch.cat((memory_domain_labels_up, domain_labels), dim=0)

            ## update mean of Ci
            mean_domain_up = CalculateMean(memory_source_features_up, memory_domain_labels_up, n_domains)

            # ## detach
            # mean_domain_up = mean_domain_up.detach()

            ## tiao
            # mlp_loss = ce_criterion(y_causal, labels_source)
            mlp_loss = ce_criterion_pro(y_causal_log_mean_softmax, labels_source)


            ## make forward pass for gnn head  ## using causal features
            ## tiao
            gnn_loss, edge_loss = causal_gcn(classifier_gnn, x_causal_feature_all, n_domains, labels_source)
            # gnn_loss, edge_loss = causal_gcn_c1(classifier_gnn, x_reconstructed_causal, n_domains, labels_source)   ## second causal graph
            # gnn_loss, edge_loss = causal_gcn_pm(classifier_gnn, x_reconstructed_causal, n_domains, labels_source)


            ### Variational inference loss
            #### Disentanglement
            MSE_res = F.mse_loss(x_reconstructed, x_original, reduction='mean')  # 重建损失，使用均方误差
            KL_fi = -0.5 * torch.mean(1 + log_var_1 - mu_1.pow(2) - log_var_1.exp())  # KL散度1
            KL_fs = -0.5 * torch.mean(1 + log_var_2 - mu_2.pow(2) - log_var_2.exp())  # KL散度2


            ### Variational inference loss
            #### x1_new ce + kl
            yc_celoss = ce_criterion(y_x1_c1, labels_source)

            Q1 = F.softmax(y_x1_c2, dim=1)
            log_Q1 = F.log_softmax(y_x1_c2, dim=1)
            # 创建均匀分布P，其概率为1/n_domains
            P1 = torch.full_like(Q1, fill_value=1.0 / n_domains)
            log_P1 = torch.log(P1)
            # 计算后向KL散度
            # yd_kl = torch.sum(Q1 * (log_Q1 - torch.log(P1)), dim=1).mean()
            yd_kl = kl_loss(log_P1, Q1)


            #### x2_new ce + kl
            yd_celoss = ce_criterion(y_x2_c2, domain_labels)

            Q2 = F.softmax(y_x2_c1, dim=1)
            log_Q2 = F.log_softmax(y_x2_c1, dim=1)
            # 创建均匀分布P，其概率为1/n_domains
            P2 = torch.full_like(Q2, fill_value=1.0 / class_num)
            log_P2 = torch.log(P2)
            # 计算后向KL散度
            # yc_kl = torch.sum(Q2 * (log_Q2 - torch.log(P2)), dim=1).mean()
            yc_kl = kl_loss(log_P2, Q2)


            kld_weight = kl_anneal_function(1, i)


            # total loss and backpropagation
            loss = config['lambda_mlp'] * mlp_loss + config['lambda_node3'] * gnn_loss + config['lambda_edge3'] * edge_loss + \
              config['lambda_kl_g'] * kld_weight * (KL_fi + KL_fs) + config['lambda_ce'] * (yc_celoss + yd_celoss) + config['lambda_kl'] * (yd_kl + yc_kl) + config['lambda_res'] * MSE_res

            loss.backward()
            optimizer.step()

            # printout train loss
            if i % 20 == 0 or i == config['source_iters'] - 1:
                log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f\tyc_celoss:%.4f\tyd_celoss:%.4f\tyd_kl:%.4f\tyc_kl:%.4f\tMSE_res:%.4f\tKL_fi:%.4f\tKL_fs:%.4f' % ( i,
                      config['source_iters'], config['lambda_mlp'] * mlp_loss.item(), config['lambda_node3'] * gnn_loss.item(), config['lambda_edge3'] * edge_loss.item(),
                      config['lambda_ce'] * yc_celoss.item(), config['lambda_ce'] * yd_celoss.item(), config['lambda_kl'] * yd_kl.item(), config['lambda_kl'] * yc_kl.item(),
                      config['lambda_res'] * MSE_res.item(), config['lambda_kl_g'] * kld_weight * KL_fi.item(), config['lambda_kl_g'] * kld_weight * KL_fs.item() )
                utils.write_logs(config, log_str)

            # evaluate network every test_interval
            if i % config['test_interval'] == config['test_interval'] - 1:
                evaluate_new(i, config, base_network, classifier_gnn, dset_loaders['target_test_causal'], mean_domain_up, list_acc)

            # if i == 5:
            #     cal_acc_tsne(i, config, base_network, classifier_gnn, dset_loaders['target_test_causal'], mean_domain_up, dset_loaders)

        else:
            raise ValueError('causal_flag cannot be recognized.')

    return base_network, classifier_gnn



def causal_gcn(classifier_gnn, x_causal_feature_all, n_domains, labels_source):

    feature_dim = x_causal_feature_all.size(1)
    x_reconstructed_causal_reshaped = x_causal_feature_all.view(n_domains, -1, feature_dim)

    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()
    ce_criterion_pro = nn.NLLLoss()

    ## 循环
    total_logits_list = []
    total_edge_sim = 0

    for domain in range(n_domains):
        # 提取每个domain对应的 classxfeature_dim 维度的矩阵
        x_reconstructed_causal_domain = x_reconstructed_causal_reshaped[domain]

        logits_gnn, edge_sim = classifier_gnn(x_reconstructed_causal_domain)

        # 累加每次循环的结果
        total_logits_list.append(logits_gnn)
        total_edge_sim += edge_sim


    average_edge_sim = total_edge_sim / n_domains
    # compute edge loss
    edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))

    ### log_mean_softmax
    total_logits_tensor = torch.stack(total_logits_list)
    y_causal_grouped_log_softmax = F.log_softmax(total_logits_tensor, dim=2)

    log_domain = torch.log(torch.tensor(n_domains))
    y_causal_log_mean_softmax = torch.logsumexp(y_causal_grouped_log_softmax, dim=0) - log_domain

    ## loss
    gnn_loss = ce_criterion_pro(y_causal_log_mean_softmax, labels_source)
    edge_loss = criterion_gedge(average_edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

    return gnn_loss, edge_loss


def causal_gcn_pm(classifier_gnn, x_reconstructed_causal, n_domains, labels_source):

    feature_dim = x_reconstructed_causal.size(1)
    x_reconstructed_causal_reshaped = x_reconstructed_causal.view(n_domains, -1, feature_dim)

    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.NLLLoss()

    ## 循环
    total_logits_gnn_p = 0
    total_edge_sim = 0

    for domain in range(n_domains):
        # 提取每个domain对应的 classxfeature_dim 维度的矩阵
        x_reconstructed_causal_domain = x_reconstructed_causal_reshaped[domain]

        logits_gnn, edge_sim = classifier_gnn(x_reconstructed_causal_domain)

        logits_gnn_p = nn.LogSoftmax(dim=1)(logits_gnn)

        # 累加每次循环的结果
        total_logits_gnn_p += logits_gnn_p
        total_edge_sim += edge_sim

    average_logits_gnn_p = total_logits_gnn_p / n_domains
    average_edge_sim = total_edge_sim / n_domains
    # compute edge loss
    edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))

    gnn_loss = ce_criterion(average_logits_gnn_p, labels_source)
    edge_loss = criterion_gedge(average_edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

    return gnn_loss, edge_loss


def causal_gcn_c1(classifier_gnn, x_reconstructed_causal, n_domains, labels_source):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    logits_gnn, edge_sim = classifier_gnn(x_reconstructed_causal)
    # compute edge loss
    edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))

    gnn_loss = ce_criterion(logits_gnn, labels_source)
    edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

    return gnn_loss, edge_loss


def adapt_target(config, base_network, classifier_gnn, dset_loaders, max_inherit_domain, mean_source, cv_source, memory_source_features, memory_source_labels, memory_domain_labels, last = False):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()
    # add random layer and adversarial network
    class_num = config['encoder']['params']['class_num']


    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() \
                     + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target_train'][max_inherit_domain])
    # set nets in train mode
    base_network.train()
    classifier_gnn.train()

    memory_source_features_up = memory_source_features
    memory_source_labels_up = memory_source_labels
    memory_domain_labels_up = memory_domain_labels

    num_samples_t = len(dset_loaders['target_train'][max_inherit_domain].dataset)
    memory_target_features = torch.zeros(num_samples_t, 256).cuda()
    memory_target_labels = torch.zeros(num_samples_t).long().cuda()
    memory_tardomain_labels = torch.zeros(num_samples_t).long().cuda()

    #——————————————————
    list_acc = []
    # —————————————————

    for i in range(config['adapt_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'][max_inherit_domain])
        batch_source = next(iter_source)
        batch_target = next(iter_target)
        inputs_source, inputs_target = batch_source['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source = batch_source['target'].to(DEVICE)
        domain_labels = batch_source['flag'].to(DEVICE)
        tardomain_labels = batch_target['flag'].to(DEVICE)

        domain_source, domain_target = batch_source['domain'].to(DEVICE), batch_target['domain'].to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source)
        features_target, logits_mlp_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        logits_mlp = torch.cat((logits_mlp_source, logits_mlp_target), dim=0)
        softmax_mlp = nn.Softmax(dim=1)(logits_mlp)
        mlp_loss = ce_criterion(logits_mlp_source, labels_source)

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features)
        gnn_loss = ce_criterion(logits_gnn[:labels_source.size(0)], labels_source)
        # compute pseudo-labels for affinity matrix by mlp classifier
        out_target_class = torch.softmax(logits_mlp_target, dim=1)
        target_score, target_pseudo_labels = out_target_class.max(1, keepdim=True)
        idx_pseudo = target_score > config['threshold']               #########  这里是做什么?
        target_pseudo_labels[~idx_pseudo] = classifier_gnn.mask_val
        # combine source labels and target pseudo labels for edge_net
        node_labels = torch.cat((labels_source, target_pseudo_labels.squeeze(dim=1)), dim=0).unsqueeze(dim=0)
        # compute source-target mask and ground truth for edge_net
        edge_gt, edge_mask = classifier_gnn.label2edge(node_labels)

        # ### addxu
        # assert not torch.isnan(features_source).any(), f"Output features_source contains NaN at iteration {i}"
        # assert not torch.isnan(features_target).any(), f"Output features_target contains NaN at iteration {i}"
        # assert not torch.isnan(logits_mlp_source).any(), f"Output logits_mlp_source contains NaN at iteration {i}"
        # assert not torch.isnan(logits_mlp_target).any(), f"Output logits_mlp_target contains NaN at iteration {i}"


        # compute edge loss
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))


        ## addxu update mean CV - step 2
        num_samples_s = len(dset_loaders['source'].dataset)
        batch_size = config['data']['source']['batch_size']
        cat_number = features_source.shape[0]
        cat_number_t = features_target.shape[0]

        if config['update_mem_flag'].startswith('true'):


            if memory_source_features_up.shape[0] < num_samples_s:

                memory_source_features_up = torch.cat((memory_source_features_up, features_source), dim=0)
                memory_source_labels_up = torch.cat((memory_source_labels_up, labels_source), dim=0)
                memory_domain_labels_up = torch.cat((memory_domain_labels_up, domain_labels), dim=0)


            if memory_source_features_up.shape[0] >= num_samples_s:

                # 首先移除 memory_source_features 和 memory_source_labels 的前 batch_size 个元素
                memory_source_features_up = memory_source_features_up[cat_number:]
                memory_source_labels_up = memory_source_labels_up[cat_number:]
                memory_domain_labels_up = memory_domain_labels_up[cat_number:]

                # 然后在尾部添加新的 features_source 和 labels_source
                memory_source_features_up = torch.cat((memory_source_features_up, features_source), dim=0)
                memory_source_labels_up = torch.cat((memory_source_labels_up, labels_source), dim=0)
                memory_domain_labels_up = torch.cat((memory_domain_labels_up, domain_labels), dim=0)

        ####  Judge whether it is the last target domain
        if last:
            _, target_p_labels = out_target_class.max(1, keepdim=True)
            target_p_labels = target_p_labels.squeeze(1)

            memory_target_features = memory_target_features[cat_number_t:]
            memory_target_labels = memory_target_labels[cat_number_t:]
            memory_tardomain_labels = memory_tardomain_labels[cat_number_t:]

            memory_target_features = torch.cat((memory_target_features, features_target), dim=0)
            memory_target_labels = torch.cat((memory_target_labels, target_p_labels), dim=0)
            memory_tardomain_labels = torch.cat((memory_tardomain_labels, tardomain_labels), dim=0)



        if config['update_mem_flag'].startswith('true'):
            mean_source_up = CalculateMean(memory_source_features_up, memory_source_labels_up, class_num)
            cv_source_up = Calculate_CV(memory_source_features_up, memory_source_labels_up, mean_source_up, class_num)
        elif config['update_mem_flag'].startswith('false'):
            mean_source_up = mean_source
            cv_source_up = cv_source
        else:
            raise ValueError('update_mem_flag cannot be recognized.')

        # ### addxu
        # assert not torch.isnan(memory_source_features_up).any(), f"Output memory_source_features_up contains NaN at iteration {i}"
        # assert not torch.isnan(memory_source_labels_up).any(), f"Output memory_source_labels_up contains NaN at iteration {i}"
        # assert not torch.isnan(mean_source_up).any(), f"Output mean_source_up contains NaN at iteration {i}"
        # assert not torch.isnan(cv_source_up).any(), f"Output cv_source_up contains NaN at iteration {i}"
        # assert not torch.isnan(edge_loss).any(), f"Output edge_loss contains NaN at iteration {i}"
        # assert not torch.isnan(features_target).any(), f"Output features_target contains NaN at iteration {i}"


        #### addxu   # *** MO-Method ***
        if config['method'] == 'MO':
            logits_gnn_target = logits_gnn[labels_source.size(0):]
            trans_loss = MO(mean_source_up, cv_source_up, features_target, logits_gnn_target, class_num)
        else:
            raise ValueError('Method cannot be recognized.')

        # total loss and backpropagation
        loss = config['lambda_adv'] * trans_loss + mlp_loss + \
               config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss: %.4f\t GNN Loss: %.4f\t Edge Loss: %.4f\t Transfer loss:%.4f' % (
                i, config["adapt_iters"], mlp_loss.item(), config['lambda_node'] * gnn_loss.item(),
                config['lambda_edge'] * edge_loss.item(), config['lambda_adv'] * trans_loss.item()
            )
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'], list_acc)

    #### cat source & last target
    if last:
        memory_source_features_up = torch.cat((memory_source_features_up, memory_target_features), dim=0)
        memory_source_labels_up = torch.cat((memory_source_labels_up, memory_target_labels), dim=0)
        memory_domain_labels_up = torch.cat((memory_domain_labels_up, memory_tardomain_labels), dim=0)

    return base_network, classifier_gnn, memory_source_features_up, memory_source_labels_up, memory_domain_labels_up


def adapt_target_cgct(config, base_network, classifier_gnn, dset_loaders, random_layer, adv_net):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() + adv_net.get_parameters() \
                     + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target_train'])
    # set nets in train mode
    base_network.train()
    classifier_gnn.train()
    adv_net.train()
    random_layer.train()

    #——————————————————
    list_acc = []
    # —————————————————

    for i in range(config['adapt_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'])

        batch_source = next(iter_source)
        batch_target = next(iter_target)
        inputs_source, inputs_target = batch_source['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source, labels_target = batch_source['target'].to(DEVICE), batch_target['target'].to(DEVICE)
        mask_target = batch_target['mask'].bool().to(DEVICE)
        domain_source, domain_target = batch_source['domain'].to(DEVICE), batch_target['domain'].to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source)
        features_target, logits_mlp_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        logits_mlp = torch.cat((logits_mlp_source, logits_mlp_target), dim=0)
        softmax_mlp = nn.Softmax(dim=1)(logits_mlp)
        # ce loss for MLP head
        mlp_loss = ce_criterion(torch.cat((logits_mlp_source, logits_mlp_target[mask_target]), dim=0),
                                torch.cat((labels_source, labels_target[mask_target]), dim=0))

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features)
        # compute pseudo-labels for affinity matrix by mlp classifier
        out_target_class = torch.softmax(logits_mlp_target, dim=1)
        target_score, target_pseudo_labels = out_target_class.max(1, keepdim=True)
        idx_pseudo = target_score > config['threshold']
        idx_pseudo = mask_target.unsqueeze(1) | idx_pseudo
        target_pseudo_labels[~idx_pseudo] = classifier_gnn.mask_val
        # combine source labels and target pseudo labels for edge_net
        node_labels = torch.cat((labels_source, target_pseudo_labels.squeeze(dim=1)), dim=0).unsqueeze(dim=0)
        # compute source-target mask and ground truth for edge_net
        edge_gt, edge_mask = classifier_gnn.label2edge(node_labels)
        # compute edge loss
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))
        # ce loss for GNN head
        gnn_loss = ce_criterion(classifier_gnn(torch.cat((features_source, features_target[mask_target]), dim=0))[0],
                                torch.cat((labels_source, labels_target[mask_target]), dim=0))

        # *** Adversarial net at work ***
        if config['method'] == 'CDAN+E':
            entropy = transfer_loss.Entropy(softmax_mlp)
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp], adv_net,
                                            entropy, networks.calc_coeff(i), random_layer, domain_input)
        elif config['method'] == 'CDAN':
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp],
                                            adv_net, None, None, random_layer, domain_input)
        else:
            raise ValueError('Method cannot be recognized.')

        # total loss and backpropagation
        loss = config['lambda_adv'] * trans_loss + mlp_loss + \
               config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss: %.4f\t GNN Loss: %.4f\t Edge Loss: %.4f\t Transfer loss:%.4f' % (
                i, config["adapt_iters"], mlp_loss.item(), config['lambda_node'] * gnn_loss.item(),
                config['lambda_edge'] * edge_loss.item(), config['lambda_adv'] * trans_loss.item()
            )
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'], list_acc)

    return base_network, classifier_gnn


def upgrade_source_domain(config, max_inherit_domain, dsets, dset_loaders, base_network, classifier_gnn):
    target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                               dataset=max_inherit_domain, transform=config['prep']['test'], domain_label=0,
                               dataset_name=config['dataset'], split='train')
    target_loader = DataLoader(target_dataset, batch_size=128,
                               num_workers=config['num_workers'], drop_last=False)
    # set networks to eval mode
    base_network.eval()
    classifier_gnn.eval()
    test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

    # print out logs for domain
    log_str = 'Adding pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
              % (max_inherit_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                 test_res['total_pseudo_labels'], len(target_loader.dataset))
    config["out_file"].write(str(log_str) + '\n\n')
    config["out_file"].flush()
    print(log_str + '\n')

    # sub sample the dataset with the chosen confident pseudo labels
    pseudo_source_dataset = ImageList(image_root=config['data_root'],
                                      image_list_root=config['data']['image_list_root'],
                                      dataset=max_inherit_domain, transform=config['prep']['source'],
                                      domain_label=0, dataset_name=config['dataset'], split='train',
                                      sample_masks=test_res['sample_masks'], pseudo_labels=test_res['pred_cls'])

    # append to the existing source list
    dsets['source'] = ConcatDataset((dsets['source'], pseudo_source_dataset))
    # create new source dataloader
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                        shuffle=True, num_workers=config['num_workers'],
                                        drop_last=True, pin_memory=True)


def upgrade_source_domain_new(config, max_inherit_domain, dsets, dset_loaders, base_network, classifier_gnn):
    target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                               dataset=max_inherit_domain, transform=config['prep']['test'], domain_label=0,
                               dataset_name=config['dataset'], split='train')
    target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                               num_workers=config['num_workers'], drop_last=False)
    # set networks to eval mode
    base_network.eval()
    classifier_gnn.eval()
    test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

    # print out logs for domain
    log_str = 'Adding pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
              % (max_inherit_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                 test_res['total_pseudo_labels'], len(target_loader.dataset))
    config["out_file"].write(str(log_str) + '\n\n')
    config["out_file"].flush()
    print(log_str + '\n')

    # sub sample the dataset with the chosen confident pseudo labels
    pseudo_source_dataset = ImageList(image_root=config['data_root'],
                                      image_list_root=config['data']['image_list_root'],
                                      dataset=max_inherit_domain, transform=config['prep']['source'],
                                      domain_label=0, dataset_name=config['dataset'], split='train',
                                      sample_masks=test_res['sample_masks'], pseudo_labels=test_res['pred_cls'])

    # append to the existing source list
    dsets['source'] = ConcatDataset((dsets['source'], pseudo_source_dataset))
    # create new source dataloader
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                        shuffle=True, num_workers=config['num_workers'],
                                        drop_last=True, pin_memory=True)


def upgrade_target_domains(config, dsets, dset_loaders, base_network, classifier_gnn, curri_iter):
    target_dsets_new = {}
    for target_domain in dsets['target_train']:
        target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                                   dataset=target_domain, transform=config['prep']['test'], domain_label=1,
                                   dataset_name=config['dataset'], split='train')
        target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                                   num_workers=config['num_workers'], drop_last=False)
        # set networks to eval mode
        base_network.eval()
        classifier_gnn.eval()
        test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

        # print out logs for domain
        log_str = 'Updating pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
                  % (target_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                     test_res['total_pseudo_labels'], len(target_loader.dataset))
        config["out_file"].write(str(log_str) + '\n\n')
        config["out_file"].flush()
        print(log_str + '\n')

        # update pseudo labels
        target_dataset_new = ImageList(image_root=config['data_root'],
                                       image_list_root=config['data']['image_list_root'],
                                       dataset=target_domain, transform=config['prep']['target'],
                                       domain_label=1, dataset_name=config['dataset'], split='train',
                                       sample_masks=test_res['sample_masks_cgct'],
                                       pseudo_labels=test_res['pred_cls'], use_cgct_mask=True)
        target_dsets_new[target_domain] = target_dataset_new

        if curri_iter == len(config['data']['target']['name']) - 1:
            # sub sample the dataset with the chosen confident pseudo labels
            target_dataset_new = ImageList(image_root=config['data_root'],
                                           image_list_root=config['data']['image_list_root'],
                                           dataset=target_domain, transform=config['prep']['source'],
                                           domain_label=0, dataset_name=config['dataset'], split='train',
                                           sample_masks=test_res['sample_masks'],
                                           pseudo_labels=test_res['pred_cls'], use_cgct_mask=False)

            # append to the existing source list
            dsets['source'] = ConcatDataset((dsets['source'], target_dataset_new))
    dsets['target_train'] = target_dsets_new

    if curri_iter == len(config['data']['target']['name']) - 1:
        # create new source dataloader
        dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                            shuffle=True, num_workers=config['num_workers'],
                                            drop_last=True, pin_memory=True)


def cal_acc_tsne(i, config, base_network, classifier_gnn, target_test_dset_dict, mean_domain_up, dset_loaders):
    base_network.eval()
    classifier_gnn.eval()

    for dset_name, test_loader in target_test_dset_dict.items():
        cal_acc_tsne11(config, test_loader, base_network, classifier_gnn, mean_domain_up, dset_name, dset_loaders)

    base_network.train()
    classifier_gnn.train()


def cal_acc_tsne11(config, test_loader, base_network, classifier_gnn, mean_domain_up, dset_name, dset_loaders):
    start_test = True
    start_test1 = True

    with torch.no_grad():

        n_domains = config['encoder_new']['params']['domain_num']

        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = next(iter_test)
            inputs = data['img'].to(DEVICE)
            domain_labels = data['flag'].to(DEVICE)
            labels = data['target'].to(DEVICE)

            ## forward pass
            x_original, mu_1, log_var_1, mu_2, log_var_2, x1_new, x2_new, y_x1_1, y_x1_2, y_x2_1, y_x2_2, x_reconstructed, x_causal_feature_all, y_causal_log_mean_softmax \
                = base_network(inputs, Ci=mean_domain_up, step=3, domainlabel=domain_labels)

            x_causal_feature_mean = x_causal_feature_all.view( n_domains, x1_new.size(0), -1 )
            x_causal_feature_mean = x_causal_feature_mean.mean(0)

            if start_test:
                all_x1_new = x1_new.float().cpu()
                all_x2_new = x2_new.float().cpu()
                all_x_causal_feature_mean = x_causal_feature_mean.float().cpu()

                all_label = labels.float()
                all_domain_labels = domain_labels.float()

                start_test = False

            else:

                all_x1_new = torch.cat((all_x1_new, x1_new.float().cpu()), 0)
                all_x2_new = torch.cat((all_x2_new, x2_new.float().cpu()), 0)
                all_x_causal_feature_mean = torch.cat((all_x_causal_feature_mean, x_causal_feature_mean.float().cpu()), 0)

                all_label = torch.cat((all_label, labels.float()), 0)
                all_domain_labels = torch.cat((all_domain_labels, domain_labels.float()), 0)


        dataNew_fi = 'CDAN' + str(config['data']['source']['name']) + '-' + str(dset_name.upper()) + '-' + 'fi.mat'   ##  gai
        dataNew_fs = 'CDAN' + str(config['data']['source']['name']) + '-' + str(dset_name.upper()) + '-' + 'fs.mat'   ##  gai
        dataNew_causal = 'CDAN' + str(config['data']['source']['name']) + '-' + str(dset_name.upper()) + '-' + 'causal.mat'   ##  gai

        all_x1_new_numpy = all_x1_new.cpu().numpy()
        all_x2_new_numpy = all_x2_new.cpu().numpy()
        all_x_causal_feature_mean_numpy = all_x_causal_feature_mean.cpu().numpy()

        all_label_numpy = all_label.cpu().numpy() + 1
        all_domain_labels_numpy = all_domain_labels.cpu().numpy() + 1

        scio.savemat( dataNew_fi, {'Zt1': all_x1_new_numpy, 'Yt1': all_label_numpy} )
        scio.savemat( dataNew_fs, {'Zt2': all_x2_new_numpy, 'Yt2': all_domain_labels_numpy} )
        scio.savemat( dataNew_causal, {'Zt3': all_x_causal_feature_mean_numpy, 'Yt3': all_label_numpy} )

        print('tsne-target have done22')


        ### source
        iter_test = iter(dset_loaders["source11"])
        for i in range(len(iter(dset_loaders["source11"]))):
            data = next(iter_test)
            inputs = data['img'].to(DEVICE)
            domain_labels = data['flag'].to(DEVICE)
            labels = data['target'].to(DEVICE)

            ## forward pass
            x_original, mu_1, log_var_1, mu_2, log_var_2, x1_new, x2_new, y_x1_1, y_x1_2, y_x2_1, y_x2_2, x_reconstructed, x_causal_feature_all, y_causal_log_mean_softmax \
                = base_network(inputs, Ci=mean_domain_up, step=3, domainlabel=domain_labels)

            x_causal_feature_mean = x_causal_feature_all.view( n_domains, x1_new.size(0), -1 )
            x_causal_feature_mean = x_causal_feature_mean.mean(0)


            if start_test1:
                all_x1_new = x1_new.float().cpu()
                all_x2_new = x2_new.float().cpu()
                all_x_causal_feature_mean = x_causal_feature_mean.float().cpu()

                all_label = labels.float()
                all_domain_labels = domain_labels.float()

                start_test1 = False

            else:

                all_x1_new = torch.cat((all_x1_new, x1_new.float().cpu()), 0)
                all_x2_new = torch.cat((all_x2_new, x2_new.float().cpu()), 0)
                all_x_causal_feature_mean = torch.cat((all_x_causal_feature_mean, x_causal_feature_mean.float().cpu()), 0)

                all_label = torch.cat((all_label, labels.float()), 0)
                all_domain_labels = torch.cat((all_domain_labels, domain_labels.float()), 0)


        dataNew_fi = 'CDAN' + str(config['data']['source']['name']) + '-' + 'fi.mat'   ##  gai
        dataNew_fs = 'CDAN' + str(config['data']['source']['name']) + '-' + 'fs.mat'   ##  gai
        dataNew_causal = 'CDAN' + str(config['data']['source']['name']) + '-' + 'causal.mat'   ##  gai


        all_x1_new_numpy = all_x1_new.cpu().numpy()
        all_x2_new_numpy = all_x2_new.cpu().numpy()
        all_x_causal_feature_mean_numpy = all_x_causal_feature_mean.cpu().numpy()

        all_label_numpy = all_label.cpu().numpy() + 1
        all_domain_labels_numpy = all_domain_labels.cpu().numpy() + 1

        scio.savemat( dataNew_fi, {'Zs1': all_x1_new_numpy, 'Ys1': all_label_numpy} )
        scio.savemat( dataNew_fs, {'Zs2': all_x2_new_numpy, 'Ys2': all_domain_labels_numpy} )
        scio.savemat( dataNew_causal, {'Zs3': all_x_causal_feature_mean_numpy, 'Ys3': all_label_numpy} )

    print('tsne-source have done22')


