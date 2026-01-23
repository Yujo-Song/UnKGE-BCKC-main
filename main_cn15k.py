from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# 设置 CuBLAS 确定性环境变量，二选一即可
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys

if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')
from os.path import join
from src.data import Data
import numpy as np
import random
import datetime
import argparse

import torch

from src.trainer import Trainer
from src.utils import *



def get_model_identifier(model):
    prefix = model
    now = datetime.datetime.now()
    date = '%02d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
    identifier = prefix + '_' +'_'+ date
    return identifier

def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)


def set_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=3407, type=int, help='Random seed.')

    # only test
    parser.add_argument('--only_test', action='store_true', help="is or not only_test.")
    parser.add_argument('--models_dir', type=str, default='./trained_models', help="the dir path where you store trained models. A new directory will be created inside it.")

    # required
    parser.add_argument('--data', type=str, default='cn15k', help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
    # optional
    parser.add_argument("--verbose", help="print detailed info for debugging",action="store_true")
    parser.add_argument('-d', '--dim', type=int, default=512, help="set dimension. default: 128")
    parser.add_argument('--epoch', type=int, default=500, help="set number of epochs. default: 500")
    parser.add_argument('--lr', type=float, default=0.0001, help="set learning rate. default: 0.001")
    parser.add_argument('--batch_size', type=int, default=4096, help="set batch size. default: 1024")
    parser.add_argument('--n_neg', type=int, default=30, help="Number of negative samples per (h,r,t). default: 10")

    # BERT相关参数
    parser.add_argument('--use_bert', type=bool, default=True, help="是否使用BERT编码器初始化embedding")
    parser.add_argument('--bert_model', type=str, default='roberta-base',help="BERT模型名称，默认bert-base-uncased")
    parser.add_argument('--freeze_bert', type=bool, default=True,help="是否冻结BERT参数。True=不微调BERT, False=微调BERT")
    parser.add_argument('--regenerate_bert', type=bool, default=True,help="是否重新生成BERT embeddings（即使缓存存在）。默认False，会优先加载缓存")
    parser.add_argument('--bert_cache_dir', type=str, default=None,help="BERT embeddings缓存目录。如果不指定，使用{save_dir}/bert_embeddings")

    # loss
    parser.add_argument('--reg_scale', type=float, default=0.001,help="The scale for regularizer (lambda) of calculate confidence. Default 0.005")

    # 上下文增强BERT参数
    parser.add_argument('--max_neighbors', type=int, default=5,help="每个实体最多使用多少个邻居三元组进行上下文增强。默认5")
    parser.add_argument('--min_weight', type=float, default=0.85,help="邻居三元组的最小权重阈值。默认0.6")

    # 聚类相关参数
    parser.add_argument('--use_clustering', type=bool, default=True,help="是否使用聚类增强模块（簇共享embedding+门控融合+对比学习）")
    parser.add_argument('--num_entity_clusters', type=int, default=70,help="实体簇数量。推荐: CN15K=70, NL27K=130, PPI5K=50")
    parser.add_argument('--num_relation_clusters', type=int, default=10,help="关系簇数量。推荐: CN15K=10, NL27K=30, PPI5K=3")
    parser.add_argument('--regenerate_clustering', type=bool, default=True,help="是否重新执行聚类（即使缓存存在）。默认False，会优先加载缓存")
    parser.add_argument('--clustering_cache_dir', type=str, default=None,help="聚类结果缓存目录。如果不指定，使用{save_dir}/clustering")

    # 对比学习参数
    parser.add_argument('--contrastive_weight', type=float, default=0.0001,help="对比学习损失的权重。默认0.0001")
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,help="对比学习温度参数。越小越关注hard negatives。默认0.1")
    parser.add_argument('--entity_contrastive_weight', type=float, default=1.0,help="实体对比学习损失的权重。默认1.0")
    parser.add_argument('--relation_contrastive_weight', type=float, default=1.0,help="关系对比学习损失的权重。默认1.0")

    # early_stop
    parser.add_argument('--early_stop', type=bool, default=True, help="early stop.")
    parser.add_argument('--early_stop_patience', type=int, default=30, help="early stop patience. default: 10")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = set_parser()  # 设置参数
    seed_everything(args.seed) # 设置随机数
    print('seed is: ', args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path to save
    identifier = get_model_identifier("unkg")
    save_dir = join(args.models_dir, args.data, identifier)  # the directory where we store this model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Trained models will be stored in: ', save_dir)

    # input files
    data_dir = join('./data', args.data)
    print('Read train.tsv from', data_dir)

    # load data
    this_data = Data(args, data_dir)
    this_data.load_data()
    this_data.record_more_data()
    # this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

    # BERT save
    args.bert_cache_dir = join(args.models_dir, args.data, 'Bert_embeddings_' + args.bert_model)
    if not os.path.exists(args.bert_cache_dir):
        os.makedirs(args.bert_cache_dir)

    args.clustering_cache_dir = join(args.models_dir, args.data, 'Cluster_embeddings')
    if not os.path.exists(args.clustering_cache_dir):
        os.makedirs(args.clustering_cache_dir)

    m_train = Trainer(args, device)
    m_train.build(this_data, save_dir)

    # Model will be trained, validated, and saved in './trained_models'
    if args.only_test:
        m_train.test("test")
    else:
        m_train.train()
    # m_train.test(filename=data_dir)