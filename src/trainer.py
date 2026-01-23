''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from os.path import join

from src import utils

import sys

if '../src' not in sys.path:
    sys.path.append('../src')

import numpy as np
import time
from src.data import BatchLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.autograd import Variable
import operator
from src.models import unKG
from src.utils import *
from tqdm import tqdm

import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save = False
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save = True


class Trainer(object):
    def __init__(self, args, device):

        self.verbose = args.verbose  # print extra information
        self.epoch = args.epoch
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.neg_per_positive = args.n_neg
        self.reg_scale = args.reg_scale
        self.lr = args.lr
        self.args = args

        self.device = device

        self.this_data = None
        # self.batchloader = None
        self.tf_parts = None
        self.file_val = ""
        self.L1 = False
        self.early_stop = self.args.early_stop
        self.early_stop_epoch = None
        self.mse_is_save = False


    def build(self, data_obj, save_dir):
        """
        All files are stored in save_dir.
        output files:
        1. tf model
        2. this_data (Data())
        3. training_loss.csv, val_loss.csv
        :param model_save: filename for model
        :param data_save: filename for self.this_data
        :return:
        """

        self.this_data = data_obj
        self.batchloader = BatchLoader(self.this_data, self.batch_size, self.neg_per_positive)
        # paths for saving
        self.save_dir = save_dir
        self.train_loss_path = join(save_dir, 'training_loss.csv')
        self.val_loss_path = join(save_dir, 'val_loss.csv')
        self.test_loss_path = join(save_dir, 'test_loss.csv')


        self.model = unKG(self.this_data.num_rels(), self.this_data.num_cons(), self.dim, self.batch_size, self.neg_per_positive, self.reg_scale, self.this_data, self.args, self.device, save_dir).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):

        print('Number of epochs: %d' % self.epoch)
        num_batch = self.this_data.triples.shape[0] // self.batch_size
        print('Number of batches per epoch: %d' % num_batch)

        train_losses = []  # [[every epoch, loss]]
        # val_losses = []  # [[saver epoch, loss]]
        # test_losses = []  # [[saver epoch, loss]]

        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
        checkpoint = self.save_dir + '/checkpoint'

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

        start = time.time()

        for epoch in range(1, self.epoch + 1):

            self.model.train()

            loss_epoch = 0.0
            confidence_loss_epoch = 0.0
            ranking_loss_epoch = 0.0
            contrastive_loss_epoch = 0.0
            regularizer_loss_epoch = 0.0

            # process batch train data to cal loss
            epoch_batches = self.batchloader.gen_batch(forever=True)

            for batch_id in tqdm(range(num_batch)):

                self.optimizer.zero_grad()

                batch = next(epoch_batches)
                A_h_index, A_r_index, A_t_index, A_w, \
                    A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, \
                    A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch

                # loss
                loss, confidence_loss, ranking_loss, contrastive_loss, regularizer_loss = self.model(A_h_index, A_r_index, A_t_index, A_w, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index)

                loss_epoch += loss.item()
                confidence_loss_epoch += confidence_loss.item()
                ranking_loss_epoch += ranking_loss.item()
                contrastive_loss_epoch += contrastive_loss.item()
                regularizer_loss_epoch += regularizer_loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            train_losses.append([epoch, loss_epoch])
            print("Loss of epoch %d = %s" % (epoch, loss_epoch), "CP Loss = %s" % confidence_loss_epoch, "LP Loss = %s" % ranking_loss_epoch, "Contrastive Loss = %s" % contrastive_loss_epoch, "Re Loss = %s" % regularizer_loss_epoch)

            self.model.eval()
            with torch.no_grad():
                result = self.metrics(model_mse= self.model, eval="val")
                for key, value in result.items():
                    print(key + ":" + str(value) + "         ")

                if epoch % 50 == 0:
                    torch.save(self.model, checkpoint + '/' + str(epoch) + '_model.pt')

                # early-stop
                if self.early_stop:
                    # use mse_pos for early_stop
                    val_loss = result["MSE_pos"]

                    early_stopping(val_loss)

                    if early_stopping.save:
                        '''Saves model when validation loss decrease.'''

                        torch.save(self.model, checkpoint + '/cp_model.pt')
                        self.mse_is_save = True


                    if early_stopping.early_stop:
                        # 触发早停条件，但是继续训练，直到结束
                        self.early_stop = False
                        self.early_stop_epoch = epoch - self.args.early_stop_patience

                        # '''Saves model when validation loss decrease.'''
                        #
                        # if not os.path.exists(checkpoint):
                        #     os.makedirs(checkpoint)
                        # torch.save(self.model, checkpoint + '/' + str(epoch) + 'KE.pt')
                        # break
        # 训练结束，保存最终的模型
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        torch.save(self.model, checkpoint + '/lp_model.pt')
        # 如果此时mse的模型还没有保存，进行保存
        if not self.mse_is_save:
            if not os.path.exists(checkpoint):
                os.makedirs(checkpoint)
            torch.save(self.model, checkpoint + '/cp_model.pt')
            self.early_stop_epoch = self.epoch

        # training time
        print('Train FINISHED.')
        time_consumed = time.time() - start
        print('Time consumed(s):', time_consumed)
        print("Early stopping, best mse epoch = %d" % self.early_stop_epoch)

        # loss graph
        self.print_loss_graph(train_losses)

        # 测试
        print("over training, start testing")
        self.model.eval()

        model_test_mse = torch.load(checkpoint + '/cp_model.pt')
        model_test_link = torch.load(checkpoint + '/lp_model.pt')
        model_test_mse.eval()
        model_test_link.eval()

        # # 保存训练后的embeddings（用于后续直接加载）
        # print("\n" + "=" * 70)
        # print("保存训练后的embeddings")
        # print("=" * 70)
        # model_test_mse.save_trained_embeddings()
        # print("=" * 70 + "\n")

        with torch.no_grad():
            filename = join('./data', self.args.data)

            result_test = self.metrics(model_test_mse, model_test_link, filename=filename, eval="test")
            self.print_(result_test)

    def metrics(self, model_mse, model_link = None, filename=None, eval="val"):

        result = dict()
        if eval == "val":
            mae_pos, mse_pos = conf_predict(self.this_data.val_triples, model_mse)
            # mae_neg, mse_neg = get_mse_neg(self.this_data.test_triples, self.this_data, model_mse,self.neg_per_positive)
            result["MSE_pos"] = mse_pos
            result["MAE_pos"] = mae_pos

        else:
            mae_pos, mse_pos = conf_predict(self.this_data.test_triples, model_mse)
            # mae_neg, mse_neg = get_mse_neg(self.this_data.test_triples, self.this_data, model_mse, self.neg_per_positive)
            # result["MSE"] = ((mse_pos+mse_neg)/2)*100
            # result["MAE"] = ((mae_pos+mae_neg)/2)*100
            result["MSE_pos"] = mse_pos
            result["MAE_pos"] = mae_pos

            pred_thres = np.arange(0, 1, 0.05)
            if self.args.data == 'ppi5k':
                threshod = 0.7
            else:
                threshod = 0.85
            P, R, f1, Acc = classify_triples(self.this_data, model_mse, threshod, pred_thres)
            # print('\n', np.max(f1), '\n', np.max(Acc), '\n')
            result['F-1'] = np.max(f1)
            result['Accu'] = np.max(Acc)

            self.this_data.load_hr_map(filename)
            hr_map = get_fixed_hr(self.this_data.hr_map, n=200)
            tr_map = get_fixed_hr(self.this_data.tr_map, n=200)
            # hr_map = self.this_data.hr_map

            print("开始链接预测评估")
            mr, mrr, wmr, wmrr, hit_1, hit_3, hit_5, hit_10, hit_20, hit_40 = link_prediction(self.this_data, model_link)


            result["MR"] = mr
            result["MRR"] = mrr
            result['hits@1'] = hit_1
            result['hits@3'] = hit_3
            result['hits@5'] = hit_5
            result['hits@10'] = hit_10
            result["WMR"] = wmr
            result["WMRR"] = wmrr
            result['WH@20'] = hit_20
            result['WH@40'] = hit_40

            mean_ndcg, mean_exp_ndcg = mean_ndcg_(hr_map, model_link, self.this_data)
            result["ndcg(linear)"] = mean_ndcg
            result["ndcg(exp)"] = mean_exp_ndcg

        return result

    def test(self, model_dir_now):
        # 测试
        print("start only testing")

        path = join(self.args.models_dir, self.args.data, model_dir_now, 'checkpoint')

        model_test_mse = torch.load(path + '/cp_model.pt')
        model_test_link = torch.load(path + '/lp_model.pt')
        model_test_mse.eval()
        model_test_link.eval()
        with torch.no_grad():
            filename = join('./data', self.args.data)

            result_test = self.metrics(model_test_mse, model_test_link, filename=filename, eval="test")
            self.print_(result_test)

    def print_(self, result):
        for key, value in result.items():
            print(key + ":" + str(value))

        # 指定要写入的文件名
        filename = self.save_dir + '/result.txt'

        # 打开文件并写入数据
        with open(filename, 'w') as file:
            for key, value in result.items():
                file.write(f'{key}: {value}\n')


    def print_loss_graph(self, epoch_loss):

        # 解压列表，获取epoch和loss
        epochs = [loss[0] for loss in epoch_loss]
        losses = [loss[1] for loss in epoch_loss]

        # 创建折线图
        plt.figure(figsize=(10, 5))  # 设置图表大小
        plt.plot(epochs, losses, linestyle='-', color='b')  # 绘制折线图，标记为圆圈，线型为实线，颜色为蓝色
        plt.title('Training Loss per Epoch')  # 图表标题
        plt.xlabel('Epoch')  # x轴标签
        plt.ylabel('Loss')  # y轴标签
        # plt.grid(True)  # 显示网格
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域

        # 保存图表到本地
        plt.savefig(self.save_dir + '/loss_chart.png')
        # 保存图表为PDF
        plt.savefig(self.save_dir + '/loss_chart.pdf')

        # 显示图表
        plt.show()
