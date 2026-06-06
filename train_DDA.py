import argparse
import csv
import os
import timeit
from collections import defaultdict

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


def build_positive_lookup(samples, labels):
    lookup = defaultdict(set)
    labels = np.asarray(labels).reshape(-1)
    for pair, label in zip(samples, labels):
        if int(label) == 1:
            lookup[int(pair[0])].add(int(pair[1]))
    return lookup


def contrastive_retrieval_loss(dr_emb, di_emb, samples, labels):
    labels = labels.flatten().bool()
    if labels.sum() == 0 or (~labels).sum() == 0:
        return dr_emb.new_tensor(0.0)

    drug_query = fn.normalize(dr_emb[samples[:, 0]], dim=-1)
    disease_query = fn.normalize(di_emb[samples[:, 1]], dim=-1)
    logits = torch.matmul(drug_query, disease_query.t())
    targets = torch.arange(samples.shape[0], device=samples.device)
    positive_logits = logits[labels]
    positive_targets = targets[labels]

    if positive_logits.shape[0] == 0:
        return dr_emb.new_tensor(0.0)
    return fn.cross_entropy(positive_logits, positive_targets)


def mine_hard_negative_samples(model, dr, di, positive_lookup, device):
    if model.retrieval_reasoner is None or model.disease_bank is None:
        return None

    negatives = []
    all_disease_ids = torch.arange(di.shape[0], device=device)
    with torch.no_grad():
        for drug_id, positive_diseases in positive_lookup.items():
            drug_emb = dr[drug_id].unsqueeze(0)
            search_k = min(di.shape[0] - 1, max(model.retrieval_top_k * 4, model.retrieval_top_k + len(positive_diseases) + 1))
            indices, _ = model.retrieval_reasoner.disease_reasoner.retriever.search(
                drug_emb,
                search_k,
                query_ids=None,
            )
            chosen = None
            for disease_id in indices[0].detach().cpu().tolist():
                if disease_id not in positive_diseases:
                    chosen = disease_id
                    break
            if chosen is None:
                mask = torch.ones(di.shape[0], dtype=torch.bool, device=device)
                if positive_diseases:
                    mask[torch.tensor(list(positive_diseases), device=device)] = False
                candidates = all_disease_ids[mask]
                if candidates.numel() > 0:
                    chosen = int(candidates[0].item())
            if chosen is not None:
                negatives.append([drug_id, chosen])

    if not negatives:
        return None
    return torch.LongTensor(negatives).to(device)


def log_retrieval_tsv(path, fold, epoch, samples, retrieval_info):
    if retrieval_info is None:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    samples_np = samples.detach().cpu().numpy()

    with open(path, 'a', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        if not file_exists:
            writer.writerow([
                'fold', 'epoch', 'sample_index', 'drug_id', 'disease_id',
                'query_type', 'neighbor_rank', 'neighbor_id',
                'similarity', 'attention_weight'
            ])

        for query_type, entity_info in [('drug', retrieval_info.drug), ('disease', retrieval_info.disease)]:
            indices = entity_info.indices.detach().cpu().numpy()
            scores = entity_info.scores.detach().cpu().numpy()
            weights = entity_info.attention_weights.detach().cpu().numpy()
            for i in range(samples_np.shape[0]):
                drug_id = int(samples_np[i, 0])
                disease_id = int(samples_np[i, 1])
                for rank, neighbor_id in enumerate(indices[i]):
                    writer.writerow([
                        fold, epoch, i, drug_id, disease_id, query_type,
                        rank, int(neighbor_id),
                        float(scores[i, rank]), float(weights[i, rank])
                    ])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--retrieval_mode', default='full', choices=['full', 'baseline', 'retrieval'], help='retrieval mode')
    parser.add_argument('--top_k', type=int, default=10, help='top_k for retrieval')
    parser.add_argument('--retrieval_heads', type=int, default=4, help='multi-head retrieval attention heads')
    parser.add_argument('--retrieval_dropout', type=float, default=0.3, help='probability of dropping retrieval context during training')
    parser.add_argument('--index_refresh_epochs', type=int, default=5, help='rebuild FAISS indexes every N epochs')
    parser.add_argument('--lambda_contrastive', type=float, default=0.05, help='contrastive retrieval loss weight')
    parser.add_argument('--use_hard_negatives', action='store_true', help='train with mined nearest unassociated disease negatives')

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
        'mode': args.retrieval_mode,
        'top_k': args.top_k,
        'use_gpu': True,
        'index_refresh': 'per_epoch',
        'index_refresh_epochs': args.index_refresh_epochs,
        'num_heads': args.retrieval_heads,
        'dropout': args.retrieval_dropout,
        'log_interpretability': True,
        'log_every_epoch': False,
    }

    split_config = {
        'mode': 'standard',
        # 'mode': 'cold_start_drug',
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
        positive_lookup = build_positive_lookup(data['X_train'][i], data['Y_train'][i])

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            should_refresh_index = (
                args.retrieval_config['mode'] != 'baseline'
                and args.retrieval_config['index_refresh'] == 'per_epoch'
                and epoch % args.retrieval_config['index_refresh_epochs'] == 0
            )
            if should_refresh_index:
                with torch.no_grad():
                    model.eval()
                    model.update_retrieval_index(
                        drdr_graph, didi_graph, drdipr_graph,
                        drug_feature, disease_feature, protein_feature
                    )
                model.train()

            model.train()
            dr_train, train_score, _ = model(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature, X_train
            )
            di_train = model.disease_bank if model.disease_bank is not None else model.encode_nodes(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature
            )[1]
            classification_loss = cross_entropy(train_score, torch.flatten(Y_train))
            contrastive_loss = contrastive_retrieval_loss(dr_train, di_train, X_train, Y_train)
            train_loss = classification_loss + args.lambda_contrastive * contrastive_loss

            if args.use_hard_negatives and args.retrieval_config['mode'] != 'baseline':
                hard_negative_samples = mine_hard_negative_samples(model, dr_train, di_train, positive_lookup, device)
                if hard_negative_samples is not None:
                    hard_negative_labels = torch.zeros(hard_negative_samples.shape[0], dtype=torch.long, device=device)
                    _, hard_negative_score, _ = model(
                        drdr_graph, didi_graph, drdipr_graph,
                        drug_feature, disease_feature, protein_feature, hard_negative_samples
                    )
                    train_loss = train_loss + cross_entropy(hard_negative_score, hard_negative_labels)

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
                    log_retrieval_tsv(log_path, i, best_epoch, X_test, retrieval_info)

            if args.retrieval_config['log_interpretability'] and args.retrieval_config['log_every_epoch']:
                log_path = os.path.join(args.result_dir, 'retrieval_logs', 'fold_{}_epoch.tsv'.format(i))
                log_retrieval_tsv(log_path, i, epoch + 1, X_test, retrieval_info)

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



