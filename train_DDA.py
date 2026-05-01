import argparse
import csv
import os
import timeit

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from data_preprocess import *
from metric import *
from model.AMNTDDA import AMNTDDA

device = torch.device('cuda')


def log_retrieval_tsv(path, fold, epoch, samples, retrieval_info, drug_number):
    if retrieval_info is None:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    indices = retrieval_info.indices.detach().cpu().numpy()
    scores = retrieval_info.scores.detach().cpu().numpy()
    weights = retrieval_info.attention_weights.detach().cpu().numpy()
    samples_np = samples.detach().cpu().numpy()

    with open(path, 'a', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        if not file_exists:
            writer.writerow([
                'fold', 'epoch', 'sample_index', 'drug_id', 'disease_id',
                'neighbor_rank', 'neighbor_global_id', 'neighbor_type', 'neighbor_local_id',
                'similarity', 'attention_weight'
            ])

        for i in range(samples_np.shape[0]):
            drug_id = int(samples_np[i, 0])
            disease_id = int(samples_np[i, 1])
            for rank, neighbor_id in enumerate(indices[i]):
                neighbor_id = int(neighbor_id)
                if neighbor_id < drug_number:
                    neighbor_type = 'drug'
                    neighbor_local = neighbor_id
                else:
                    neighbor_type = 'disease'
                    neighbor_local = neighbor_id - drug_number
                writer.writerow([
                    fold, epoch, i, drug_id, disease_id,
                    rank, neighbor_id, neighbor_type, neighbor_local,
                    float(scores[i, rank]), float(weights[i, rank])
                ])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    retrieval_config = {
        'mode': 'full',
        'top_k': 10,
        'query_type': 'mlp',
        'use_gpu': True,
        'index_refresh': 'per_epoch',
        'log_interpretability': True,
        'log_every_epoch': False,
    }

    split_config = {
        'mode': 'standard',
    }

    args.retrieval_config = retrieval_config
    args.split_config = split_config

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    if args.split_config['mode'] == 'standard':
        data = k_fold(data, args)
    else:
        data = cold_start_k_fold(data, args, args.split_config['mode'])

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []
    ACCs, PRECs, RECs, F1s, MCCs = [], [], [], [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = AMNTDDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            if args.retrieval_config['mode'] != 'baseline' and args.retrieval_config['index_refresh'] == 'per_epoch':
                with torch.no_grad():
                    model.eval()
                    model.update_retrieval_index(
                        drdr_graph, didi_graph, drdipr_graph,
                        drug_feature, disease_feature, protein_feature
                    )
                model.train()

            model.train()
            _, train_score, _ = model(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature, X_train
            )
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score, retrieval_info = model(
                    drdr_graph, didi_graph, drdipr_graph,
                    drug_feature, disease_feature, protein_feature, X_test
                )

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

                if args.retrieval_config['log_interpretability'] and retrieval_info is not None:
                    log_path = os.path.join(args.result_dir, 'retrieval_logs', 'fold_{}_best.tsv'.format(i))
                    log_retrieval_tsv(log_path, i, best_epoch, X_test, retrieval_info, args.drug_number)

            if args.retrieval_config['log_interpretability'] and args.retrieval_config['log_every_epoch']:
                log_path = os.path.join(args.result_dir, 'retrieval_logs', 'fold_{}_epoch.tsv'.format(i))
                log_retrieval_tsv(log_path, i, epoch + 1, X_test, retrieval_info, args.drug_number)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCs.append(best_accuracy)
        PRECs.append(best_precision)
        RECs.append(best_recall)
        F1s.append(best_f1)
        MCCs.append(best_mcc)

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('Accuracy:', ACCs)
    print('Mean Accuracy:', np.mean(ACCs), '(', np.std(ACCs), ')')

    print('Precision:', PRECs)
    print('Mean Precision:', np.mean(PRECs), '(', np.std(PRECs), ')')

    print('Recall:', RECs)
    print('Mean Recall:', np.mean(RECs), '(', np.std(RECs), ')')

    print('F1:', F1s)
    print('Mean F1:', np.mean(F1s), '(', np.std(F1s), ')')

    print('MCC:', MCCs)
    print('Mean MCC:', np.mean(MCCs), '(', np.std(MCCs), ')')



