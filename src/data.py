"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import pandas as pd
from os.path import join

import torch

from src import utils
from collections import defaultdict as ddict


class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''

    def __init__(self, args, data_dir):
        self.args = args


        # ID映射字典（从entity_id.csv和relation_id.csv加载）
        self.entity_id2text = {}  # {id: text}
        self.entity_text2id = {}  # {text: id}
        self.relation_id2text = {}  # {id: text}
        self.relation_text2id = {}  # {text: id}

        # 实体和关系的数量（从ID映射文件确定）
        self.num_entities = 0
        self.num_relations = 0

        # 数据集
        self.triples = np.array([0])  # training dataset
        self.val_triples = np.array([0])  # validation dataset
        self.test_triples = np.array([0])  # test dataset

        self.hr_map = {} # ndcg test
        self.tr_map = {}

        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)

        # (h,r,t) tuples(int), no w
        # set containing train, val, test (for negative sampling).
        self.triples_record = set([])
        self.weights = np.array([0])

        # 用于上下文增强BERT - 存储每个实体的邻居信息
        self.entity_neighbors = None  # {entity_id: [(relation_id, neighbor_id, is_head, weight)]}
        self.relation_cooccurrence = None  # {relation_id: {other_relation_id: count}} - 关系共现信息
        self.relation_triplets = None  # {relation_id: [(head_id, tail_id, weight)]} - 每个关系包含的三元组

        self.neg_triples = np.array([0])

        # test for rank
        self.head_candidate = set([])
        self.tail_candidate = set([])

        # recorded for tf_parts
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.L1 = False

        # 数据目录
        self.data_dir = data_dir

    def load_id_mappings(self, entity_file, relation_file):
        """
        从entity_id.csv和relation_id.csv加载ID映射

        支持不同数据集的格式差异：
        - cn15k: 无header, 格式为 entity_text,id 和 relation_text,id
        - nl27k: 有header, 格式为 id,entity_text 和 id,relation_text
        - ppi5k: 有header, 格式为 entity_text,id 和 id,relation_text
        """
        if self.args.data == "cn15k":
            # cn15k: 无header, entity_text,id
            df_entity = pd.read_csv(entity_file, header=None, names=['entity_text', 'entity_id'])
            df_relation = pd.read_csv(relation_file, header=None, names=['relation_text', 'relation_id'])

        elif self.args.data == "nl27k":
            # nl27k: 有header, id,entity_text
            df_entity = pd.read_csv(entity_file, header=0)  # 跳过header
            # 检测列名并重命名
            if 'id' in df_entity.columns:
                df_entity = df_entity.rename(columns={'id': 'entity_id', 'entity string': 'entity_text'})

            df_relation = pd.read_csv(relation_file, header=0)  # 跳过header
            # 检测列名并重命名
            if 'rid' in df_relation.columns:
                df_relation = df_relation.rename(columns={'rid': 'relation_id', 'relation': 'relation_text'})

        elif self.args.data == "ppi5k":
            # ppi5k: 有header, entity_text,id 和 id,relation_text
            df_entity = pd.read_csv(entity_file, header=0)  # 跳过header
            df_entity = df_entity.rename(columns={'entity string': 'entity_text', 'id': 'entity_id'})

            df_relation = pd.read_csv(relation_file, header=0)  # 跳过header
            df_relation = df_relation.rename(columns={'id': 'relation_id', 'rel string': 'relation_text'})

        # 加载实体映射
        for _, row in df_entity.iterrows():
            entity_text = str(row['entity_text']).strip()
            entity_id = int(row['entity_id'])
            self.entity_id2text[entity_id] = entity_text
            self.entity_text2id[entity_text] = entity_id

        self.num_entities = len(self.entity_id2text)
        print(f"  ✓ 加载了 {self.num_entities} 个实体映射")

        # 加载关系映射
        for _, row in df_relation.iterrows():
            relation_text = str(row['relation_text']).strip()
            relation_id = int(row['relation_id'])
            self.relation_id2text[relation_id] = relation_text
            self.relation_text2id[relation_text] = relation_id

        self.num_relations = len(self.relation_id2text)
        print(f"  ✓ 加载了 {self.num_relations} 个关系映射\n")

    def load_triples(self, filename, splitter='\t', line_end='\n'):
        '''
        加载三元组数据

        重要：train.tsv中的ID是整数，直接对应entity_id.csv中的ID
        '''
        if self.num_entities == 0 or self.num_relations == 0:
            raise RuntimeError("请先调用load_id_mappings()加载ID映射！")

        triples = []
        # print(f"正在加载三元组: {filename}")

        for line in open(filename, encoding='utf-8'):
            line = line.rstrip(line_end).split(splitter)

            if len(line) < 4:
                continue

            # 直接将字符串转换为整数ID（不重新映射）
            h = int(line[0])
            r = int(line[1])
            t = int(line[2])
            w = float(line[3])

            # 验证ID的有效性
            if h < 0 or h >= self.num_entities:
                raise ValueError(f"无效的头实体ID: {h}（有效范围：[0, {self.num_entities-1}]）")
            if t < 0 or t >= self.num_entities:
                raise ValueError(f"无效的尾实体ID: {t}（有效范围：[0, {self.num_entities-1}]）")
            if r < 0 or r >= self.num_relations:
                raise ValueError(f"无效的关系ID: {r}（有效范围：[0, {self.num_relations-1}]）")

            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)

            triples.append([h, r, t, w])
            self.triples_record.add((h, r, t))

        triples = np.array(triples)
        # print(f"✓ 加载了 {len(triples)} 个三元组\n")
        return triples
    
    def get_head_candidates(self):
        """获取数据集中出现过的头实体候选集合"""
        return list(self.head_candidate)
    
    def get_tail_candidates(self):
        """获取数据集中出现过的尾实体候选集合"""
        return list(self.tail_candidate)

    def load_data(self, splitter='\t', line_end='\n'):
        """
        加载所有数据文件
        """

        file_train = join(self.data_dir, 'train.tsv')  # training data
        file_psl = join(self.data_dir, 'softlogic.tsv')  # training data
        file_val = join(self.data_dir, 'val.tsv')  # validation data
        file_test = join(self.data_dir, 'test.tsv')

        # 加载实体映射
        entity_file = join(self.data_dir, 'entity_id.csv')
        # 加载关系映射
        relation_file = join(self.data_dir, 'relation_id.csv')


        self.load_id_mappings(entity_file, relation_file)

        # 加载三元组数据
        self.triples = self.load_triples(file_train, splitter, line_end)
        self.val_triples = self.load_triples(file_val, splitter, line_end)
        self.test_triples = self.load_triples(file_test, splitter, line_end)

        # 只有重新生成BERT才构建实体和关系的邻居信息
        if self.args.regenerate_bert:
            # 构建实体邻居信息（用于上下文增强BERT）
            self._build_entity_neighbors()

            # 构建关系三元组索引（用于关系的上下文增强BERT）
            self._build_relation_triplets()

    def _build_entity_neighbors(self):
        """
        构建每个实体的邻居信息，用于上下文增强的BERT编码

        邻居信息格式：
        entity_neighbors[entity_id] = [
            (relation_id, neighbor_entity_id, is_head, weight),
            ...
        ]

        is_head=True: 当前实体是头实体，即 entity -> relation -> neighbor
        is_head=False: 当前实体是尾实体，即 neighbor -> relation -> entity
        """
        print("正在构建实体邻居信息...")

        # 初始化邻居字典 - 使用num_entities确保所有实体都有条目
        self.entity_neighbors = {i: [] for i in range(self.num_entities)}

        # 初始化关系共现字典
        self.relation_cooccurrence = {i: ddict(int) for i in range(self.num_relations)}

        # 遍历所有训练三元组
        for h, r, t, w in self.triples:
            h, r, t = int(h), int(r), int(t)
            w = float(w)

            # 为头实体添加邻居信息
            # (relation, neighbor, is_head=True, weight)
            self.entity_neighbors[h].append((r, t, True, w))

            # 为尾实体添加邻居信息
            # (relation, neighbor, is_head=False, weight)
            self.entity_neighbors[t].append((r, h, False, w))

        # 为每个实体按权重排序邻居（高权重的邻居更重要）
        for entity_id in self.entity_neighbors:
            self.entity_neighbors[entity_id].sort(key=lambda x: x[3], reverse=True)

        # # 统计关系共现（同一个实体的不同关系）
        # for entity_id, neighbors in self.entity_neighbors.items():
        #     relations = [r for r, _, _, _ in neighbors]
        #     # 计算关系共现
        #     for i, r1 in enumerate(relations):
        #         for r2 in relations[i+1:]:
        #             self.relation_cooccurrence[r1][r2] += 1
        #             self.relation_cooccurrence[r2][r1] += 1

        # 统计信息
        total_neighbors = sum(len(neighbors) for neighbors in self.entity_neighbors.values())
        avg_neighbors = total_neighbors / len(self.entity_neighbors) if len(self.entity_neighbors) > 0 else 0

        print(f"✓ 实体邻居信息构建完成")
        print(f"  - 总实体数: {len(self.entity_neighbors)}")
        print(f"  - 总邻居数: {total_neighbors}")
        print(f"  - 平均邻居数: {avg_neighbors:.2f}")

    def _build_relation_triplets(self):
        """
        构建每个关系包含的三元组索引，用于关系的上下文增强BERT编码

        关系三元组格式：
        relation_triplets[relation_id] = [
            (head_id, tail_id, weight),
            ...
        ]
        所有三元组按置信度降序排列
        """
        print("正在构建关系三元组索引...")

        # 初始化关系三元组字典
        self.relation_triplets = {i: [] for i in range(self.num_relations)}

        # 遍历所有训练三元组
        for h, r, t, w in self.triples:
            h, r, t = int(h), int(r), int(t)
            w = float(w)

            # 为关系添加三元组
            self.relation_triplets[r].append((h, t, w))

        # 为每个关系的三元组按置信度降序排列
        for relation_id in self.relation_triplets:
            self.relation_triplets[relation_id].sort(key=lambda x: x[2], reverse=True)

        # 统计信息
        total_triplets = sum(len(triplets) for triplets in self.relation_triplets.values())
        avg_triplets = total_triplets / len(self.relation_triplets) if len(self.relation_triplets) > 0 else 0

        print(f"✓ 关系三元组索引构建完成")
        print(f"  - 总关系数: {len(self.relation_triplets)}")
        print(f"  - 总三元组数: {total_triplets}")
        print(f"  - 平均每个关系的三元组数: {avg_triplets:.2f}\n")

    def get_entity_context(self, entity_id, max_neighbors=5, min_weight=0.7):
        """
        获取实体的上下文信息（邻居三元组）

        Args:
            entity_id: 实体ID
            max_neighbors: 最多返回多少个邻居，默认5
            min_weight: 最小权重阈值，只返回权重>=min_weight的邻居，默认0.5

        Returns:
            context_triples: [(relation_id, neighbor_id, is_head, weight), ...]
        """
        if self.entity_neighbors is None:
            return []

        neighbors = self.entity_neighbors.get(entity_id, [])

        # 过滤低权重邻居
        filtered_neighbors = [n for n in neighbors if n[3] >= min_weight]

        # 返回前max_neighbors个
        return filtered_neighbors[:max_neighbors]

    # def get_relation_context(self, relation_id, top_k=3):
    #     """
    #     获取关系的上下文信息（经常与其共现的其他关系）
    #
    #     Args:
    #         relation_id: 关系ID
    #         top_k: 返回top-k个共现关系
    #
    #     Returns:
    #         cooccurring_relations: [(other_relation_id, count), ...]
    #     """
    #     if self.relation_cooccurrence is None:
    #         return []
    #
    #     cooccurrences = self.relation_cooccurrence.get(relation_id, {})
    #
    #     # 按共现次数排序
    #     sorted_cooccurrences = sorted(cooccurrences.items(), key=lambda x: x[1], reverse=True)
    #
    #     return sorted_cooccurrences[:top_k]

    def get_relation_triplets(self, relation_id, max_triplets=5, min_weight=0.7):
        """
        获取包含指定关系的三元组作为上下文（用于关系的置信度感知编码）

        从预构建的索引中查询，已经按置信度降序排列

        Args:
            relation_id: 关系ID
            max_triplets: 最多返回多少个三元组（实际上是传给调用方做进一步过滤用的）
            min_weight: 最小权重阈值（设为0.0表示先获取所有，由调用方过滤）

        Returns:
            triplets: [(head_id, tail_id, weight), ...] 包含该关系的三元组列表（已按置信度降序）
        """
        if self.relation_triplets is None:
            return []

        # 直接从索引中查询（已经按置信度降序排列）
        all_triplets = self.relation_triplets.get(relation_id, [])

        # 过滤低权重邻居
        filtered_neighbors = [n for n in all_triplets if n[2] >= min_weight]

        # 返回前max_neighbors个
        return filtered_neighbors[:max_triplets]



    # add more triples to self.triples_record to 'filt' negative sampling
    def record_more_data(self, splitter='\t', line_end='\n'):
        more_filt = [join(self.data_dir, 'val.tsv'), join(self.data_dir, 'test.tsv')]
        for filename in more_filt:
            for line in open(filename, encoding='utf-8'):
                line = line.rstrip(line_end).split(splitter)
                if len(line) < 3:
                    continue
                h = int(line[0])
                r = int(line[1])
                t = int(line[2])

                # 验证ID有效性
                if 0 <= h < self.num_entities and 0 <= r < self.num_relations and 0 <= t < self.num_entities:
                    self.triples_record.add((h, r, t))

    def load_hr_map(self, filename, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        """
        with open(join(filename, 'test.tsv'), encoding='utf-8') as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = int(line[0])
                r = int(line[1])
                t = int(line[2])
                w = float(line[3])

                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

                if self.tr_map.get(t) == None:
                    self.tr_map[t] = {}
                if self.tr_map[t].get(r) == None:
                    self.tr_map[t][r] = {h: w}
                else:
                    self.tr_map[t][r][h] = w

                self.head_candidate.add(h)
                self.tail_candidate.add(t)

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        supplement_t_files = ['train.tsv', 'val.tsv','test.tsv']
        for file in supplement_t_files:
            with open(join(filename, file), encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = int(line[0])
                    r = int(line[1])
                    t = int(line[2])
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

                    if t in self.tr_map and r in self.tr_map[t]:
                        self.tr_map[t][r][h] = w
    def num_cons(self):
        '''Returns number of entities.

        This means all entities have index that 0 <= index < num_cons().
        '''
        return self.num_entities

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        '''
        return self.num_relations

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.relation_text2id.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.relation_id2text.get(rel_index)

    def con_str2index(self, con_str):
        '''For entity `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.entity_text2id.get(con_str)

    def con_index2str(self, con_index):
        '''For entity `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.entity_id2text.get(con_index)

    def rel(self):
        return np.array(range(self.num_rels()))

    def corrupt_pos(self, triple, pos):
        """
        :param triple: [h, r, t]
        :param pos: index position to replace (0 for h, 2 fot t)
        :return: [h', r, t] or [h, r, t']
        """
        hit = True
        res = None
        while hit:
            res = np.copy(triple)
            samp = np.random.randint(self.num_cons())
            while samp == triple[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            # # debug
            # if tuple(res) in self.triples_record:
            #     print('negative sampling: rechoose')
            #     print(res)
            if tuple(res) not in self.triples_record:
                hit = False
        return res

    # bernoulli negative sampling
    def corrupt(self, triple, neg_per_positive, tar=None):
        """
        :param triple: [h r t]
        :param tar: 't' or 'h'
        :return: np.array [[h,r,t1],[h,r,t2],...]
        """
        # print("array.shape:", res.shape)
        if tar == 't':
            position = 2
        elif tar == 'h':
            position = 0
        res = [self.corrupt_pos(triple, position) for i in range(neg_per_positive)]
        return np.array(res)

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    # bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar=None):
        res = np.array([self.corrupt(triple, neg_per_positive, tar) for triple in t_batch])
        return res

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        # print("Save data object as", filename)

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)

    def save_meta_table(self, save_dir):
        """
        save index-entity, index-relation table to file.
        File: idx_concept.csv, idx_relation.csv
        """
        idx_con_path = join(save_dir, 'idx_concept.csv')
        df_con = pd.DataFrame({
            'index': list(self.entity_id2text.keys()),
            'concepts': list(self.entity_id2text.values())
        })
        df_con.sort_values(by='index').to_csv(idx_con_path, index=None)

        idx_rel_path = join(save_dir, 'idx_relation.csv')
        df_rel = pd.DataFrame({
            'index': list(self.relation_id2text.keys()),
            'relations': list(self.relation_id2text.values())
        })
        df_rel.sort_values(by='index').to_csv(idx_rel_path, index=None)


class BatchLoader():
    def __init__(self, data_obj, batch_size, neg_per_positive):
        self.this_data = data_obj  # Data() object
        self.shuffle = True
        self.batch_size = batch_size
        self.neg_per_positive = neg_per_positive

    def gen_batch(self, forever=False, shuffle=True):
        """
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]),
                                           axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:,
                                                                                                       2].astype(
                    int), batch[:, 3]
                hrt_batch = batch[:, 0:3].astype(int)

                # all_neg_hn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                # all_neg_tn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

                neg_hn_batch, neg_rel_hn_batch, \
                neg_t_batch, neg_h_batch, \
                neg_rel_tn_batch, neg_tn_batch \
                    = self.corrupt_batch(h_batch, r_batch, t_batch)

                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(
                    np.int64), w_batch.astype(
                    np.float32), \
                    neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                    neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                    neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)

            if not forever:
                break

    def corrupt_batch(self, h_batch, r_batch, t_batch):
        N = self.this_data.num_cons()  # number of entities

        neg_hn_batch = np.random.randint(0, N, size=(
        self.batch_size, self.neg_per_positive))  # random index without filtering
        neg_rel_hn_batch = np.tile(r_batch, (self.neg_per_positive, 1)).transpose()  # copy
        neg_t_batch = np.tile(t_batch, (self.neg_per_positive, 1)).transpose()

        neg_h_batch = np.tile(h_batch, (self.neg_per_positive, 1)).transpose()
        neg_rel_tn_batch = neg_rel_hn_batch
        neg_tn_batch = np.random.randint(0, N, size=(self.batch_size, self.neg_per_positive))

        return neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch
