import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease
from model.retrieval_reasoning import RetrievalReasoner

device = torch.device('cuda')


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.retrieval_config = getattr(args, 'retrieval_config', {})
        self.retrieval_mode = self.retrieval_config.get('mode', 'baseline')
        self.retrieval_top_k = self.retrieval_config.get('top_k', 10)
        self.retrieval_query_type = self.retrieval_config.get('query_type', 'sum')
        self.retrieval_use_gpu = self.retrieval_config.get('use_gpu', True)
        self.retrieval_refresh = self.retrieval_config.get('index_refresh', 'per_forward')
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout)
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer-1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        self.retrieval_reasoner = None
        self.embedding_bank = None
        if self.retrieval_mode != 'baseline':
            self.retrieval_reasoner = RetrievalReasoner(
                dim=args.gt_out_dim * 2,
                top_k=self.retrieval_top_k,
                mode=self.retrieval_mode,
                query_type=self.retrieval_query_type,
                use_gpu=self.retrieval_use_gpu,
            )


    def encode_nodes(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:self.args.disease_number+self.args.drug_number, :]

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        return dr, di

    def _update_retrieval_index(self, dr: torch.Tensor, di: torch.Tensor) -> None:
        if self.retrieval_reasoner is None:
            return
        self.embedding_bank = torch.cat([dr, di], dim=0)
        self.retrieval_reasoner.build_index(self.embedding_bank)

    def update_retrieval_index(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature):
        dr, di = self.encode_nodes(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature)
        self._update_retrieval_index(dr, di)

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        dr, di = self.encode_nodes(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature)

        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])
        retrieval_info = None

        if self.retrieval_mode != 'baseline':
            if self.retrieval_refresh == 'per_forward' or self.embedding_bank is None:
                self._update_retrieval_index(dr, di)

            query_dr = dr[sample[:, 0]]
            query_di = di[sample[:, 1]]
            context, retrieval_info = self.retrieval_reasoner(query_dr, query_di, self.embedding_bank)
            drdi_embedding = drdi_embedding + context

        output = self.mlp(drdi_embedding)

        return dr, output, retrieval_info

