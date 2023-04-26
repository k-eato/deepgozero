import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from torch_utils import FastTensorDataLoader
import seaborn as sns
from torch.distributions import uniform


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgozero_boxel_256_nf3.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgozero_boxel_256_nf3.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    loss_func = nn.BCELoss()
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)
    
    normal_forms, relations, zero_classes = load_normal_forms(go_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    # for k, v in normal_forms.items():
    #     normal_forms[k] = th.LongTensor(v).to(device)

    net = DGELModel(n_iprs, n_terms, n_zeros, n_rels, device).to(device)
    print(net)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    nf_bs = 64
    nb_train_data = 0
    for key, val in normal_forms.items():
        nb_train_data = max(len(val), nb_train_data)
    train_steps = int(math.ceil(nb_train_data / (1.0 * nf_bs)))
    nf_generator = Generator(normal_forms, steps=train_steps)
    nf_generator.reset()

    best_loss = 10000.0
    if not load:
        print('Training the model')
        roc_auc_list = []
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            lmbda = 0.1
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    nf_next, nf_labels = nf_generator.next()
                    el_loss = net.go_embed.el_loss(nf_next)
                    total_loss = loss + el_loss
                    train_loss += loss.detach().item()
                    train_elloss = el_loss.detach().item()
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                roc_auc_list.append(roc_auc)
                print(f'Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

            print('EL Loss', train_elloss)
            if epoch > 30:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

            scheduler.step()
            
        # Save and plot AUC
        df = pd.DataFrame(roc_auc_list, columns=['auc'])
        df['epoch'] = range(len(roc_auc_list))
        chart = sns.lineplot(data=df, x='epoch', y='auc')
        chart.set(xlabel='Epoch', ylabel='AUROC')
        fig = chart.get_figure()
        fig.savefig(f'{data_root}/{ont}/auc_graph.png')
        roc_auc_list = [ '%.4f' % elem for elem in roc_auc_list ]
        auc_file = f'{data_root}/{ont}/validation_auc.txt'
        auc_out = open(auc_file, 'w')
        auc_out.write('\n'.join(str(i) for i in roc_auc_list))
        auc_out.close()

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, n_terms)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        
    preds = list(preds)
    # Propagate scores using ontology structure
    for i, scores in enumerate(preds):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = scores[j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                scores[terms_dict[go_id]] = score

    test_df['preds'] = preds

    test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def load_normal_forms(go_file, terms_dict):
    zclasses = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': []}

    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                rel = 'SubClassOf'
                data['nf1'].append((get_index(go1), get_rel_index(rel), get_index(go2)))
            elif left.find('and') != -1: # C and D SubClassOf E
                go1, go2 = left.split(' and ')
                go3 = right
                form = 'nf2'
                if go3 == 'Nothing':
                    form = 'disjoint'
                data[form].append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find('some') != -1:  # R some C SubClassOf D
                rel, go1 = left.split(' some ')
                go2 = right
                data['nf4'].append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find('some') != -1: # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(' some ')
                data['nf3'].append((get_index(go1), get_rel_index(rel), get_index(go2)))

                
    # Check if TOP in zclasses and insert if it is not there
    if 'owl:Thing' not in zclasses:
        zclasses['owl:Thing'] = len(zclasses) + len(terms_dict)
    #changing by adding sub classes of train_data ids to prot_ids
    prot_ids = []
    class_keys = list(zclasses.keys())
    for go_id in terms_dict.keys():
        prot_ids.append(terms_dict[go_id])
    for go_id in zclasses.keys():
        prot_ids.append(zclasses[go_id])

    prot_ids = np.array(prot_ids)
    
    
    # Add corrupted triples nf3
    n_classes = len(zclasses)
    data['nf3_neg'] = []
    for c, r, d in data['nf3']:
        x = np.random.choice(prot_ids)
        while x == c:
            x = np.random.choice(prot_ids)
            
        y = np.random.choice(prot_ids)
        while y == d:
             y = np.random.choice(prot_ids)
        data['nf3_neg'].append((c, r,x))
        data['nf3_neg'].append((y, r, d))
        
    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([zclasses['owl:Thing'],])
    data['nf3_neg'] = np.array(data['nf3_neg'])
    
                            
    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]
    
    return data, relations, zclasses


class Generator(object):
    def __init__(self, data, batch_size=128, steps=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = steps
        self.start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.steps:
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf2_index = np.random.choice(
                self.data['nf2'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
            dis_index = np.random.choice(
                self.data['disjoint'].shape[0], self.batch_size)
            # top_index = np.random.choice(
            #     self.data['top'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf2 = self.data['nf2'][nf2_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            dis = self.data['disjoint'][dis_index]
            # top = self.data['top'][top_index]
            nf3_neg = self.data['nf3_neg'][nf3_neg_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return (th.from_numpy(np.array([nf1, nf2, nf3, nf4, dis, nf3_neg])).long(), th.from_numpy(labels))
        else:
            self.reset()
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf2_index = np.random.choice(
                self.data['nf2'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
            dis_index = np.random.choice(
                self.data['disjoint'].shape[0], self.batch_size)
            # top_index = np.random.choice(
            #     self.data['top'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf2 = self.data['nf2'][nf2_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            dis = self.data['disjoint'][dis_index]
            # top = self.data['top'][top_index]
            nf3_neg = self.data['nf3_neg'][nf3_neg_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return (th.from_numpy(np.array([nf1, nf2, nf3, nf4, dis, nf3_neg])).long(), th.from_numpy(labels))


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.BatchNorm1d(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


class BoxEL(nn.Module):
    
    def __init__(self, vocab_size, relation_size, embed_dim, min_init_value, delta_init_value, relation_init_value, scaling_init_value):
        super(BoxEL, self).__init__()
        min_embedding = self.init_concept_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_concept_embedding(vocab_size, embed_dim, delta_init_value)
        relation_embedding = self.init_concept_embedding(relation_size, embed_dim, relation_init_value)
        scaling_embedding = self.init_concept_embedding(relation_size, embed_dim, scaling_init_value)
        self.temperature = 1.0
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        self.relation_embedding = nn.Parameter(relation_embedding)
        self.scaling_embedding = nn.Parameter(scaling_embedding)

        print(self.min_embedding.shape)
        
        self.gumbel_beta = 1.0
        self.scale = 1.0
        self.eps = 1e-8

    def forward(self, go_terms):
        return self.min_embedding[go_terms]

    def el_loss(self, data):
        
        nf1_min_0 = self.min_embedding[data[0][:, 0]]
        nf1_min_2 = self.min_embedding[data[0][:, 2]]
        nf1_delta_0 = self.delta_embedding[data[0][:, 0]]
        nf1_delta_2 = self.delta_embedding[data[0][:, 2]]
        nf1_max_0 = nf1_min_0 + th.exp(nf1_delta_0)
        nf1_max_2 = nf1_min_2 + th.exp(nf1_delta_2)
        
        boxes1 = Box(nf1_min_0, nf1_max_0)
        boxes2 = Box(nf1_min_2, nf1_max_2)
        
        nf1_loss, nf1_reg_loss = self.nf1_loss(boxes1, boxes2)
        
        nf2_min_0 = self.min_embedding[data[1][:, 0]]
        nf2_min_1 = self.min_embedding[data[1][:, 1]]
        nf2_min_2 = self.min_embedding[data[1][:, 2]]
        nf2_delta_0 = self.delta_embedding[data[1][:, 0]]
        nf2_delta_1 = self.delta_embedding[data[1][:, 1]]
        nf2_delta_2 = self.delta_embedding[data[1][:, 2]]
        nf2_max_0 = nf2_min_0 + th.exp(nf2_delta_0)
        nf2_max_1 = nf2_min_1 + th.exp(nf2_delta_1)
        nf2_max_2 = nf2_min_2 + th.exp(nf2_delta_2)
        
        boxes1 = Box(nf2_min_0, nf2_max_0)
        boxes2 = Box(nf2_min_1, nf2_max_1)
        boxes3 = Box(nf2_min_2, nf2_max_2)
        
        nf2_loss,nf2_reg_loss = self.nf2_loss(boxes1, boxes2, boxes3)
        
        nf3_min_0 = self.min_embedding[data[2][:, 0]]
        nf3_min_2 = self.min_embedding[data[2][:, 2]]
        nf3_delta_0 = self.delta_embedding[data[2][:, 0]]
        nf3_delta_2 = self.delta_embedding[data[2][:, 2]]
        nf3_max_0 = nf3_min_0 + th.exp(nf3_delta_0)
        nf3_max_2 = nf3_min_2 + th.exp(nf3_delta_2)
        relation = self.relation_embedding[data[2][:,1]]
        scaling = self.scaling_embedding[data[2][:,1]]
        
        boxes1 = Box(nf3_min_0, nf3_max_0)
        boxes2 = Box(nf3_min_2, nf3_max_2)
        
        nf3_loss,nf3_reg_loss = self.nf3_loss(boxes1, relation, scaling, boxes2)
        
        nf4_min_1 = self.min_embedding[data[3][:, 1]]
        nf4_min_2 = self.min_embedding[data[3][:, 2]]
        nf4_delta_1 = self.delta_embedding[data[3][:, 1]]
        nf4_delta_2 = self.delta_embedding[data[3][:, 2]]
        nf4_max_1 = nf4_min_1 + th.exp(nf4_delta_1)
        nf4_max_2 = nf4_min_2 + th.exp(nf4_delta_2)
        relation = self.relation_embedding[data[3][:,0]]
        scaling = self.scaling_embedding[data[3][:,0]]
        
        boxes1 = Box(nf4_min_1, nf4_max_1)
        boxes2 = Box(nf4_min_2, nf4_max_2)
        
        nf4_loss,nf4_reg_loss = self.nf4_loss(relation, scaling, boxes1, boxes2)

        disjoint_min_0 = self.min_embedding[data[4][:, 0]]
        disjoint_min_1 = self.min_embedding[data[4][:, 1]]
        disjoint_delta_0 = self.delta_embedding[data[4][:, 0]]
        disjoint_delta_1 = self.delta_embedding[data[4][:, 1]]
        disjoint_max_0 = disjoint_min_0 + th.exp(disjoint_delta_0)
        disjoint_max_1 = disjoint_min_1 + th.exp(disjoint_delta_1)

        boxes1 = Box(disjoint_min_0, disjoint_max_0)
        boxes2 = Box(disjoint_min_1, disjoint_max_1)
        disjoint_loss,disjoint_reg_loss = self.disjoint_loss(boxes1, boxes2)
        
        nf3_neg_min_0 = self.min_embedding[data[5][:, 0]]
        nf3_neg_min_2 = self.min_embedding[data[5][:, 2]]
        nf3_neg_delta_0 = self.delta_embedding[data[5][:, 0]]
        nf3_neg_delta_2 = self.delta_embedding[data[5][:, 2]]
        nf3_neg_max_0 = nf3_neg_min_0 + th.exp(nf3_neg_delta_0)
        nf3_neg_max_2 = nf3_neg_min_2 + th.exp(nf3_neg_delta_2)
        
        relation = self.relation_embedding[data[5][:,1]]
        scaling = self.scaling_embedding[data[5][:,1]]
        
        boxes1 = Box(nf3_neg_min_0, nf3_neg_max_0)
        boxes2 = Box(nf3_neg_min_2, nf3_neg_max_2)
        
        nf3_neg_loss, nf3_neg_reg_loss = self.nf3_neg_loss(boxes1, relation, scaling, boxes2)
        
        all_min = self.min_embedding
        all_delta = self.delta_embedding
        all_max = all_min+th.exp(all_delta)
        boxes = Box(all_min, all_max)
        reg_loss = self.l2_side_regularizer(boxes, log_scale=True)
        total_loss = nf1_loss.sum() + nf2_loss.sum() + nf3_loss.sum() + nf4_loss.sum() + disjoint_loss.sum() + nf1_reg_loss + nf2_reg_loss + nf3_reg_loss + nf4_reg_loss + disjoint_reg_loss
        return total_loss

    def get_cond_probs(self, data):
        nf3_min = self.min_embedding[data[:,[0,2]]]
        nf3_delta = self.delta_embedding[data[:,[0,2]]]
        nf3_max = nf3_min+th.exp(nf3_delta)
        
        relation = self.relation_embedding[data[:,1]]
        
        boxes1 = Box(nf3_min[:, 0, :], nf3_max[:, 0, :])
        boxes2 = Box(nf3_min[:, 1, :], nf3_max[:, 1, :])
        
        log_intersection = th.log(th.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box2 = th.log(th.clamp(self.volumes(boxes2), 1e-10, 1e4))
        return th.exp(log_intersection-log_box2)
        
    def volumes(self, boxes):
        return F.softplus(boxes.delta_embed, beta=self.temperature).prod(1)

    def intersection(self, boxes1, boxes2):
        intersections_min = th.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = th.min(boxes1.max_embed, boxes2.max_embed)
        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box
    
    def inclusion_loss(self, boxes1, boxes2):
        log_intersection = th.log(th.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box1 = th.log(th.clamp(self.volumes(boxes1), 1e-10, 1e4))
        
        return 1-th.exp(log_intersection-log_box1)
    
    def nf1_loss(self, boxes1, boxes2):
        return self.inclusion_loss(boxes1, boxes2), self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True)
        
    def nf2_loss(self, boxes1, boxes2, boxes3):
        inter_box = self.intersection(boxes1, boxes2)
        return self.inclusion_loss(inter_box, boxes3), self.l2_side_regularizer(inter_box, log_scale=True) + self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True) + self.l2_side_regularizer(boxes3, log_scale=True)
    
    def nf3_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed*(scaling + self.eps) + relation
        trans_max = boxes1.max_embed*(scaling + self.eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        return self.inclusion_loss(trans_boxes, boxes2), self.l2_side_regularizer(trans_boxes, log_scale=True) + self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True) 
    
    def nf4_loss(self, relation, scaling, boxes1, boxes2):
        trans_min = (boxes1.min_embed - relation)/(scaling + self.eps)
        trans_max = (boxes1.max_embed - relation)/(scaling + self.eps)
        trans_boxes = Box(trans_min, trans_max)
        return self.inclusion_loss(trans_boxes, boxes2), self.l2_side_regularizer(trans_boxes, log_scale=True) + self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True) 
        
    def disjoint_loss(self, boxes1, boxes2):
        log_intersection = th.log(th.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_boxes1 = th.log(th.clamp(self.volumes(boxes1), 1e-10, 1e4))
        log_boxes2 = th.log(th.clamp(self.volumes(boxes2), 1e-10, 1e4))
        union = log_boxes1 + log_boxes2
        return th.exp(log_intersection-union), self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True)
        
    def nf3_neg_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed*(scaling + self.eps) + relation
        trans_max = boxes1.max_embed*(scaling + self.eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        return 1-self.inclusion_loss(trans_boxes, boxes2), self.l2_side_regularizer(trans_boxes, log_scale=True) + self.l2_side_regularizer(boxes1, log_scale=True) + self.l2_side_regularizer(boxes2, log_scale=True) 
        
    def init_concept_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed

    def l2_side_regularizer(self, box, log_scale: bool = True):
        """Applies l2 regularization on all sides of all boxes and returns the sum.
        """
        min_x = box.min_embed 
        delta_x = box.delta_embed  

        if not log_scale:
            return th.mean(delta_x ** 2)
        else:
            return  th.mean(F.relu(min_x + delta_x - 1 + self.eps )) +F.relu(th.norm(min_x, p=2)-1)


class DGELModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=256, embed_dim=256, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.nb_zero_gos = nb_zero_gos
        input_length = nb_iprs
        net = []
        net.append(MLPBlock(input_length, hidden_dim))
        net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        self.net = nn.Sequential(*net)

        # ELEmbeddings
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.hasFuncIndex = th.LongTensor([nb_rels]).to(device)

        # BoxEL
        self.go_embed = BoxEL(nb_gos + nb_zero_gos, nb_rels + 1, embed_dim, [1e-4, 0.2], [-0.1, 0], [-0.1,0.1], [0.9,1.1]).to(device)

        
    def forward(self, features):
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        # logits = self.go_embed.func_prediction(go_embed, x)

        hasFuncGO = go_embed * self.go_embed.scaling_embedding[self.hasFuncIndex] + self.go_embed.relation_embedding[self.hasFuncIndex]
        
        # role_assertion_loss = th.norm(trans_go-prot_embed,p=2, dim=1,keepdim=True)
        x = th.matmul(x, hasFuncGO.T)
        
        logits = th.sigmoid(x)
        
        return logits

    def predict_zero(self, features, data):
        x = self.net(features)
        go_embed = self.go_embed(data)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(data).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits
    
    
def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

def get_smooth_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32) + 0.05
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 0.9
    return data, labels


if __name__ == '__main__':
    main()
