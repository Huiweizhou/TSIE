import os
import pdb
import random
import sys
import dgl
import numpy as np
import torch
import torchmetrics.functional as MF
import torch.nn as nn
import time
from TSIE import tsie
from iter_data import NeighborSampler
from tqdm import tqdm
from utils import get_args


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    dgl.random.seed(seed)
    dgl.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


class Trainer:
    def __init__(self):
        self.time_steps = 0
        self.train_pairs = None
        self.eval_pairs = None
        self.test_transductive_pairs = None
        self.test_inductive_pairs = None
        self.test_full_pairs = None
        self.graphs = None
        self.train_co_data = None
        self.eval_co_data = None
        self.test_co_data = None
        self.data_loaders = None
        self.shuffle = False
        self.criterion = nn.BCEWithLogitsLoss()
        # self.rng = np.random.RandomState(seed=glo_seed)
        self.config, _ = get_args()
        self.data_path = self.config.data_path + '/' + str(self.config.year_start) + '-' + str(
            self.config.year_end) + '/'
        print(self.config)
        self.device = self.config.device
        self.cmd = '_'.join([
            str(self.config.batch_size),
            str(self.config.epoch),
            str(self.config.learning_rate),
            str(self.config.sample_size_1),
            str(self.config.sample_size_2),
            str(self.config.year_start),
            str(self.config.year_end),
            str(self.config.dropout_rate)
        ])
        if self.config.backbone == "Sage":
            self.cmd = self.config.agg_type + '_' + self.cmd
        self.load_data()

    def load_data(self):
        pairs_path = os.path.join(self.data_path, 'pairs')
        graphs_path = os.path.join(self.data_path, 'graphs')
        # co_data_path = os.path.join(self.data_path, 'co_data')
        self.train_pairs = torch.load(os.path.join(pairs_path, 'train.pt'))
        self.test_full_pairs = torch.load(os.path.join(pairs_path, 'test_full.pt'))
        self.test_transductive_pairs = torch.load(os.path.join(pairs_path, 'test_transductive.pt'))
        self.test_inductive_pairs = torch.load(os.path.join(pairs_path, 'test_inductive.pt'))
        self.eval_pairs = torch.load(os.path.join(pairs_path, 'eval.pt'))
        self.graphs, _ = dgl.load_graphs(os.path.join(graphs_path, 'graphs.bin'))
        self.time_steps = len(self.graphs)


    def set_batch_sampler(self, nodes, sample_size=None):
        """
        :param nodes: train, test_full, test_inductive, test_transductive, eval
        :return: NOne
        """
        assert (self.config.sample_size_1 != 0)
        if self.config.sample_size_2 != 0:
            sampler = NeighborSampler([self.config.sample_size_1, self.config.sample_size_2])
        else:
            sampler = NeighborSampler([self.config.sample_size_1])
        time_steps = len(self.graphs)
        data_loaders = []
        # for i in range(time_steps):
        #     loader = dgl.dataloading.DataLoader(
        #         self.graphs[i], nodes, sampler,
        #         device=self.device, batch_size=self.config.batch_size,
        #         shuffle=False, num_workers=1, drop_last=False
        #     )
        #     data_loaders.append(iter(loader))
        for i in range(time_steps):
            loader = dgl.dataloading.DataLoader(
                self.graphs[i], nodes, sampler,
                device=self.device, batch_size=self.config.batch_size,
                shuffle=False, num_workers=0, drop_last=False,
            )
            data_loaders.append(iter(loader))
        return data_loaders

    @staticmethod
    def get_one_batch_data(data_loader):
        res = []
        for i in range(len(data_loader)):
            try:
                res.append(next(data_loader[i]))
            except StopIteration:
                return 'End'
        return res

    def logging(self, msg=None):
        if msg: print(msg)
        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)
        with open(self.log_file, 'a+', encoding='utf-8') as f:
            f.write(msg)

    def eval(self, model):
        model.eval()
        #print('Start Evaluate')
        eval_nodes_i = self.eval_pairs[:, 0]
        eval_nodes_j = self.eval_pairs[:, 1]
        eval_labels = self.eval_pairs[:, 2]
        eval_loaders_i = self.set_batch_sampler(eval_nodes_i)
        eval_loaders_j = self.set_batch_sampler(eval_nodes_j)
        preds = []
        with tqdm(total=len(self.eval_pairs), ncols=80) as pbar:
            while True:
                cur_eval_data_i = self.get_one_batch_data(eval_loaders_i)
                cur_eval_data_j = self.get_one_batch_data(eval_loaders_j)
                assert type(cur_eval_data_i) == type(cur_eval_data_j)
                if isinstance(cur_eval_data_i[0], str) and isinstance(cur_eval_data_j[0], str):
                    break

                outs = model(cur_eval_data_i, cur_eval_data_j)
                preds.append(outs[-1].detach())
                pbar.update(self.config.batch_size)
        #print('Calculate Evaluate Scores')

        preds = torch.cat(preds)
        eval_labels = torch.Tensor(eval_labels <= self.time_steps)
        eval_labels = eval_labels.int().to(self.device)

        eval_loss = self.criterion(preds, eval_labels.float())
        preds = torch.sigmoid(preds)
        acc, auc = MF.accuracy(preds, eval_labels), MF.auroc(preds, eval_labels)
        f1, f1_mac = MF.f1_score(preds, eval_labels), MF.f1_score(preds, eval_labels, average='macro', num_classes=2,
                                                                  multiclass=True)

        self.logging(
            'Eval Results: eval loss: {:.5f}, f1: {:.5f}, f1_mac: {:.5f}, auc: {:.5f}\n'.format(
                eval_loss, f1, f1_mac, auc
            )
        )
        del preds, eval_labels
        model.train()
        return f1

    def loss(self, a, b):
        return self.criterion(a, b)

    def train(self):
        # if self.shuffle: self.rng.shuffle(self.train_pairs)
        train_nodes_i = self.train_pairs[:, 0]
        train_nodes_j = self.train_pairs[:, 1]
        train_labels = self.train_pairs[:, 2]
        model = None
        if self.config.model == 'TSIE':
            model = tsie(
                self.config.input_dim,
                self.config.hidden_dim,
                self.config.output_dim,
                self.config.rnn_wnd,
                self.config.attn_wnd,
                self.time_steps,
                self.config.agg_type,
                self.config.dropout_rate,
                self.config.nhead,
                self.config.num_layers,
                self.config.nhead_2,
                self.config.num_layers_2,
                self.config.dropout_trans,
                self.config.device
            )
            self.log_file = os.path.join(
                self.config.log_path,
                self.cmd + '_' + str(self.config.num_layers) + '_' + str(self.config.nhead) + '_' + str(
                    self.config.attn_wnd)
            )
            self.model_file = os.path.join(
                self.config.model_save_path,
                self.cmd + '_' + str(self.config.num_layers) + '_' + str(self.config.nhead) + '_' + str(
                    self.config.attn_wnd) + '.pt'
            )
        model = model.to(self.device)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5, eta_min=1e-9, last_epoch=-1)
        best_eval_f1 = -1.0
        best_eval_epoch = 0
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S \n", time.localtime())
        self.logging(cur_time)
        self.logging('-' * 70 + '\n')
        for epoch in range(self.config.epoch):
            labels_idx = 0
            loss_list = []
            data_loaders_i = self.set_batch_sampler(train_nodes_i)
            data_loaders_j = self.set_batch_sampler(train_nodes_j)
            with tqdm(total=len(self.train_pairs), ncols=80, desc='train epoch {}'.format(epoch)) as pbar:
                while True:
                    cur_labels = train_labels[labels_idx:labels_idx + self.config.batch_size]
                    cur_data_i = self.get_one_batch_data(data_loaders_i)
                    cur_data_j = self.get_one_batch_data(data_loaders_j)
                    assert type(cur_data_i) == type(cur_data_j)
                    if isinstance(cur_data_i[0], str) and isinstance(cur_data_j[0], str): break

                    outs = model(cur_data_i, cur_data_j)  # [time_steps, batch_size]
                    labels = cur_labels
                    labels = torch.Tensor(labels).to(self.device)
                    losses = []
                    for i, out in enumerate(outs):  # 遍历每个时间节点
                        if 'static' in self.config.model:
                            t = self.time_steps
                        else:
                            t = i
                        ls = self.criterion(out, (labels <= t).float())  # loss_ce
                        losses.append(ls)
                    loss = torch.mean(torch.stack(losses))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    lr_step.step()
                    loss_list.append(loss.item())
                    labels_idx += self.config.batch_size
                    pbar.update(self.config.batch_size)
            self.logging('epoch: {}, Train Loss: {:.5f} \n'.format(epoch, np.mean(loss_list)))
            eval_f1 = self.eval(model)
            if eval_f1 > best_eval_f1:
                best_eval_f1 = eval_f1
                best_eval_epoch = epoch
                if not os.path.exists(self.config.model_save_path):
                    os.makedirs(self.config.model_save_path)
                torch.save(model, self.model_file)
        model = torch.load(self.model_file)
        start_t = time.time()
        self.test(model, self.config.test_type)
        end_t = time.time()
        self.logging('best eval epoch: {} \n'.format(best_eval_epoch))
        # self.logging('Test {} Pairs \n'.format(self.config.test_type))
        self.logging('Test time cost: {:.2f} seconds \n'.format(end_t - start_t))

    def test(self, model, test_type):
        """
        :param test_type: test_full, test_inductive, test_transductive
        :return:
        """
        model = model.eval()
        test_pairs = None
        if test_type == 'full':
            test_pairs = self.test_full_pairs
        elif test_type == 'transductive':
            test_pairs = self.test_transductive_pairs
        elif test_type == 'inductive':
            test_pairs = self.test_inductive_pairs
        # else: raise f'Not a valid test_type: {test_type}'
        with torch.no_grad():
            test_nodes_i = test_pairs[:, 0]
            test_nodes_j = test_pairs[:, 1]
            test_labels = test_pairs[:, 2]
            test_loaders_i = self.set_batch_sampler(test_nodes_i)
            test_loaders_j = self.set_batch_sampler(test_nodes_j)
            preds = []
            with tqdm(total=len(test_pairs), ncols=80) as pbar:
                while True:
                    cur_test_data_i = self.get_one_batch_data(test_loaders_i)
                    cur_test_data_j = self.get_one_batch_data(test_loaders_j)
                    assert type(cur_test_data_i) == type(cur_test_data_j)
                    if isinstance(cur_test_data_i[0], str) and isinstance(cur_test_data_j[0], str):
                        break
                    outs = model(cur_test_data_i, cur_test_data_j)
                    preds.append(outs[-1])
                    pbar.update(self.config.batch_size)
            preds = torch.sigmoid(torch.cat(preds))
            test_labels = torch.Tensor(test_labels <= self.time_steps)
            test_labels = test_labels.int().to(self.device)
            acc = MF.accuracy(preds, test_labels)
            auc = MF.auroc(preds, test_labels)
            f1 = MF.f1_score(preds, test_labels)
            f1_mac = MF.f1_score(preds, test_labels, average='macro', num_classes=2, multiclass=True)
            precision = MF.precision(preds, test_labels)
            recall = MF.recall(preds, test_labels)
            self.logging('Test results: \n')
            self.logging(
                'precision: {:.5f}, recall: {:.5f}, f1: {:.5f},'
                'f1_mac: {:.5f}, auc: {:.5f}, acc: {:.5f} \n'.format(
                    precision, recall, f1, f1_mac, auc, acc
                )
            )


if __name__ == '__main__':
    glo_seed = 2025
    seed_everything(glo_seed)
    Trainer2 = Trainer()
    Trainer2.train()
