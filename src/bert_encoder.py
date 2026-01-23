
"""
BERT Encoder for Knowledge Graph Embeddings
模仿UBERT模型，使用BERT对知识图谱三元组进行编码
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
import os
from tqdm import tqdm


class BERTEncoder(nn.Module):
    """
    BERT编码器类，用于将知识图谱三元组转换为embedding

    主要功能：
    1. 加载实体和关系的ID到文本的映射
    2. 将三元组(h, r, t)的ID转换为文本
    3. 使用BERT模型编码三元组，获取实体和关系的embedding
    """

    def __init__(self, data_obj, model_name='bert-base-uncased', device='cuda', freeze_bert=True):
        """
        初始化BERT编码器

        Args:
            data_obj: Data对象，包含已加载的实体和关系映射
            model_name: BERT模型名称，默认使用bert-base-uncased
            device: 计算设备，'cuda' 或 'cpu'
            freeze_bert: 是否冻结BERT参数，默认为True
        """
        super(BERTEncoder, self).__init__()

        self.device = device

        # 直接使用Data对象中已加载的映射，避免重复加载
        self.entity_id2text = data_obj.entity_id2text
        self.entity_text2id = data_obj.entity_text2id
        self.relation_id2text = data_obj.relation_id2text
        self.relation_text2id = data_obj.relation_text2id

        self.num_entities = len(self.entity_id2text)
        self.num_relations = len(self.relation_id2text)

        # 加载BERT tokenizer和模型
        print(f"正在加载BERT模型: {model_name}")
        path = join('./bert_file', model_name)
        if model_name == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained(path)
            self.bert_model = BertModel.from_pretrained(path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(path)

        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.bert_model.to(device)
        self.bert_model.eval()  # 设置为评估模式

        self.bert_dim = self.bert_model.config.hidden_size  # BERT的输出维度，默认768
        print(f"BERT embedding维度: {self.bert_dim}")

    def ids_to_text(self, entity_ids, relation_ids):
        """
        将实体和关系的ID转换为文本

        Args:
            entity_ids: 实体ID列表或单个ID
            relation_ids: 关系ID列表或单个ID

        Returns:
            entity_texts: 实体文本列表
            relation_texts: 关系文本列表
        """
        # 处理单个ID的情况
        if isinstance(entity_ids, int):
            entity_ids = [entity_ids]
        if isinstance(relation_ids, int):
            relation_ids = [relation_ids]

        entity_texts = [self.entity_id2text.get(int(eid), f"entity_{eid}") for eid in entity_ids]
        relation_texts = [self.relation_id2text.get(int(rid), f"relation_{rid}") for rid in relation_ids]

        return entity_texts, relation_texts

    def encode_triplet_batch(self, head_ids, relation_ids, tail_ids, max_length=128):
        """
        批量编码三元组，返回实体和关系的embedding

        Args:
            head_ids: 头实体ID张量, shape [batch_size] 或 [batch_size, neg_num]
            relation_ids: 关系ID张量, shape [batch_size] 或 [batch_size, neg_num]
            tail_ids: 尾实体ID张量, shape [batch_size] 或 [batch_size, neg_num]
            max_length: BERT输入的最大长度

        Returns:
            head_embeddings: 头实体的embedding, shape [batch_size, bert_dim] 或 [batch_size, neg_num, bert_dim]
            relation_embeddings: 关系的embedding, shape [batch_size, bert_dim] 或 [batch_size, neg_num, bert_dim]
            tail_embeddings: 尾实体的embedding, shape [batch_size, bert_dim] 或 [batch_size, neg_num, bert_dim]
        """
        # 检测输入的维度
        if len(head_ids.shape) == 1:
            # 正样本模式: [batch_size]
            return self._encode_positive_batch(head_ids, relation_ids, tail_ids, max_length)
        elif len(head_ids.shape) == 2:
            # 负样本模式: [batch_size, neg_num]
            return self._encode_negative_batch(head_ids, relation_ids, tail_ids, max_length)
        else:
            raise ValueError(f"不支持的输入维度: {head_ids.shape}")

    def _encode_positive_batch(self, head_ids, relation_ids, tail_ids, max_length=128):
        """
        编码正样本批次

        Args:
            head_ids: [batch_size]
            relation_ids: [batch_size]
            tail_ids: [batch_size]

        Returns:
            head_embeddings: [batch_size, bert_dim]
            relation_embeddings: [batch_size, bert_dim]
            tail_embeddings: [batch_size, bert_dim]
        """
        batch_size = head_ids.shape[0]

        # 转换为列表
        head_ids_list = head_ids.cpu().numpy().tolist()
        relation_ids_list = relation_ids.cpu().numpy().tolist()
        tail_ids_list = tail_ids.cpu().numpy().tolist()

        # 转换为文本
        head_texts, _ = self.ids_to_text(head_ids_list, [0] * len(head_ids_list))
        _, relation_texts = self.ids_to_text([0] * len(relation_ids_list), relation_ids_list)
        tail_texts, _ = self.ids_to_text(tail_ids_list, [0] * len(tail_ids_list))

        # 构建三元组句子: "[CLS] head [SEP] relation [SEP] tail [SEP]"
        sentences = []
        for h_text, r_text, t_text in zip(head_texts, relation_texts, tail_texts):
            # 使用空格分隔，确保BERT能够正确tokenize
            sentence = f"{h_text} {r_text} {t_text}"
            sentences.append(sentence)

        # 使用BERT tokenizer编码
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # 移动到设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # 通过BERT模型
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [batch_size, seq_length, bert_dim]
            last_hidden_state = outputs.last_hidden_state

        # 提取head, relation, tail的embedding
        # 策略：使用对应token位置的embedding
        head_embeddings = []
        relation_embeddings = []
        tail_embeddings = []

        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])

            # 找到每个部分的token位置（简化版本：使用平均pooling）
            # [CLS] token在位置0
            # 这里使用简单的策略：前1/3用于head，中间1/3用于relation，后1/3用于tail
            seq_len = attention_mask[i].sum().item()

            if seq_len > 3:
                # 分三段
                head_end = max(1, seq_len // 3)
                rel_end = max(head_end + 1, 2 * seq_len // 3)

                # 对每个部分的token进行平均pooling
                head_emb = last_hidden_state[i, 1:head_end, :].mean(dim=0)
                rel_emb = last_hidden_state[i, head_end:rel_end, :].mean(dim=0)
                tail_emb = last_hidden_state[i, rel_end:seq_len, :].mean(dim=0)
            else:
                # 序列太短，使用[CLS] token
                cls_emb = last_hidden_state[i, 0, :]
                head_emb = cls_emb
                rel_emb = cls_emb
                tail_emb = cls_emb

            head_embeddings.append(head_emb)
            relation_embeddings.append(rel_emb)
            tail_embeddings.append(tail_emb)

        # 堆叠为张量
        head_embeddings = torch.stack(head_embeddings)  # [batch_size, bert_dim]
        relation_embeddings = torch.stack(relation_embeddings)  # [batch_size, bert_dim]
        tail_embeddings = torch.stack(tail_embeddings)  # [batch_size, bert_dim]

        return head_embeddings, relation_embeddings, tail_embeddings

    def _encode_negative_batch(self, head_ids, relation_ids, tail_ids, max_length=128):
        """
        编码负样本批次

        Args:
            head_ids: [batch_size, neg_num]
            relation_ids: [batch_size, neg_num]
            tail_ids: [batch_size, neg_num]

        Returns:
            head_embeddings: [batch_size, neg_num, bert_dim]
            relation_embeddings: [batch_size, neg_num, bert_dim]
            tail_embeddings: [batch_size, neg_num, bert_dim]
        """
        batch_size, neg_num = head_ids.shape

        # 展平为 [batch_size * neg_num]
        head_ids_flat = head_ids.reshape(-1)
        relation_ids_flat = relation_ids.reshape(-1)
        tail_ids_flat = tail_ids.reshape(-1)

        # 调用正样本编码函数
        head_emb_flat, rel_emb_flat, tail_emb_flat = self._encode_positive_batch(
            head_ids_flat, relation_ids_flat, tail_ids_flat, max_length
        )

        # 重塑回 [batch_size, neg_num, bert_dim]
        head_embeddings = head_emb_flat.reshape(batch_size, neg_num, -1)
        relation_embeddings = rel_emb_flat.reshape(batch_size, neg_num, -1)
        tail_embeddings = tail_emb_flat.reshape(batch_size, neg_num, -1)

        return head_embeddings, relation_embeddings, tail_embeddings

    def get_all_entity_embeddings(self, batch_size=32):
        """
        获取所有实体的BERT embedding（用于初始化实体embedding表）

        Args:
            batch_size: 批处理大小

        Returns:
            entity_embeddings: [num_entities, bert_dim]
        """
        print(f"正在为 {self.num_entities} 个实体生成BERT embedding...")

        all_embeddings = []

        # 分批处理所有实体
        for start_idx in tqdm(range(0, self.num_entities, batch_size)):
            end_idx = min(start_idx + batch_size, self.num_entities)
            entity_ids = list(range(start_idx, end_idx))

            # 获取实体文本
            entity_texts, _ = self.ids_to_text(entity_ids, [0] * len(entity_ids))

            # 使用BERT编码
            encoded = self.tokenizer(
                entity_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                # 使用[CLS] token的embedding作为实体embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu())

        # 拼接所有batch
        entity_embeddings = torch.cat(all_embeddings, dim=0)

        print(f"完成! 实体embedding shape: {entity_embeddings.shape}")
        return entity_embeddings

    def get_all_relation_embeddings(self, batch_size=32):
        """
        获取所有关系的BERT embedding（用于初始化关系embedding表）

        Args:
            batch_size: 批处理大小

        Returns:
            relation_embeddings: [num_relations, bert_dim]
        """
        print(f"正在为 {self.num_relations} 个关系生成BERT embedding...")

        all_embeddings = []

        # 分批处理所有关系
        for start_idx in range(0, self.num_relations, batch_size):
            end_idx = min(start_idx + batch_size, self.num_relations)
            relation_ids = list(range(start_idx, end_idx))

            # 获取关系文本
            _, relation_texts = self.ids_to_text([0] * len(relation_ids), relation_ids)

            # 使用BERT编码
            encoded = self.tokenizer(
                relation_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                # 使用[CLS] token的embedding作为关系embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu())

        # 拼接所有batch
        relation_embeddings = torch.cat(all_embeddings, dim=0)

        print(f"完成! 关系embedding shape: {relation_embeddings.shape}")
        return relation_embeddings

    def get_context_enhanced_embeddings(self, data, batch_size=32, max_neighbors=5, min_weight=0.6,
                                       temperature=1.0, alpha=0.7):
        """
        获取置信度感知的上下文增强实体和关系embedding（用于聚类初始化）

        实现论文中的置信度感知语义编码模块（公式4.1-4.10）：
        1. 硬过滤：根据置信度阈值过滤低质量邻居
        2. 降序采样：按置信度降序采样top-K邻居
        3. 分别编码：核心语义和邻居三元组分别编码
        4. 置信度加权：使用softmax对邻居进行加权聚合
        5. 核心-上下文融合：α * 核心 + (1-α) * 上下文

        Args:
            data: Data对象，包含entity_neighbors信息
            batch_size: 批处理大小，默认32
            max_neighbors: 每个实体最多使用多少个邻居，默认5
            min_weight: 邻居三元组的最小权重阈值（硬过滤），默认0.6
            temperature: 温度系数τ，用于softmax加权，默认1.0
            alpha: 核心语义融合系数，默认0.7（强调实体本身）

        Returns:
            entity_embeddings: [num_entities, bert_dim] - 置信度感知的实体embedding
            relation_embeddings: [num_relations, bert_dim] - 置信度感知的关系embedding
        """
        print("=" * 70)
        print(f"正在生成置信度感知的BERT embedding")
        print(f"  - 硬过滤阈值: {min_weight}")
        print(f"  - 最多邻居数: {max_neighbors}")
        print(f"  - 温度系数τ: {temperature}")
        print(f"  - 融合系数α: {alpha}")
        print("=" * 70)

        all_entity_embeddings = []

        # 分批处理所有实体
        for start_idx in range(0, self.num_entities, batch_size):
            end_idx = min(start_idx + batch_size, self.num_entities)
            entity_ids = list(range(start_idx, end_idx))

            batch_entity_embeddings = []

            for entity_id in entity_ids:
                # 获取实体文本
                entity_text = self.entity_id2text.get(entity_id, f"entity_{entity_id}")

                # 步骤1: 硬过滤 + 降序采样
                # 获取所有邻居，并按置信度过滤
                sampled_neighbors = data.get_entity_context(entity_id, max_neighbors=max_neighbors, min_weight=min_weight)

                # # 硬过滤：保留置信度 >= min_weight 的邻居
                # filtered_neighbors = [n for n in all_neighbors if n[3] >= min_weight]
                #
                # # 降序采样：按置信度排序，取top-K
                # filtered_neighbors = sorted(filtered_neighbors, key=lambda x: x[3], reverse=True)
                # sampled_neighbors = filtered_neighbors[:max_neighbors]

                # 步骤2: 分别编码核心语义（公式4.3）
                # 编码实体本身的核心语义
                core_encoded = self.tokenizer(
                    entity_text,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors='pt'
                )
                core_input_ids = core_encoded['input_ids'].to(self.device)
                core_attention_mask = core_encoded['attention_mask'].to(self.device)

                with torch.no_grad():
                    core_outputs = self.bert_model(input_ids=core_input_ids, attention_mask=core_attention_mask)
                    # 核心语义embedding（使用[CLS] token）
                    e_core = core_outputs.last_hidden_state[:, 0, :].squeeze(0)  # [bert_dim]

                if len(sampled_neighbors) == 0:
                    # 没有邻居，只使用核心语义
                    entity_embedding = e_core
                else:
                    # 步骤3: 编码邻居三元组
                    neighbor_embeddings = []
                    neighbor_weights = []

                    for rel_id, neighbor_id, is_head, weight in sampled_neighbors:
                        rel_text = self.relation_id2text.get(rel_id, f"relation_{rel_id}")
                        neighbor_text = self.entity_id2text.get(neighbor_id, f"entity_{neighbor_id}")

                        # 构建三元组文本
                        if is_head:
                            # 当前实体是头实体: entity -> relation -> neighbor
                            triplet_text = f"{entity_text} {rel_text} {neighbor_text}"
                        else:
                            # 当前实体是尾实体: neighbor -> relation -> entity
                            triplet_text = f"{neighbor_text} {rel_text} {entity_text}"

                        # 编码邻居三元组
                        neighbor_encoded = self.tokenizer(
                            triplet_text,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors='pt'
                        )
                        neighbor_input_ids = neighbor_encoded['input_ids'].to(self.device)
                        neighbor_attention_mask = neighbor_encoded['attention_mask'].to(self.device)

                        with torch.no_grad():
                            neighbor_outputs = self.bert_model(
                                input_ids=neighbor_input_ids,
                                attention_mask=neighbor_attention_mask
                            )
                            # 邻居embedding（使用[CLS] token）
                            e_neighbor = neighbor_outputs.last_hidden_state[:, 0, :].squeeze(0)  # [bert_dim]

                        neighbor_embeddings.append(e_neighbor)
                        neighbor_weights.append(weight)

                    # 步骤4: 置信度加权聚合
                    # 使用softmax计算归一化权重
                    neighbor_weights_tensor = torch.tensor(neighbor_weights, dtype=torch.float32, device=self.device)
                    # 温度缩放的softmax
                    normalized_weights = torch.nn.functional.softmax(neighbor_weights_tensor / temperature, dim=0)

                    # 加权聚合邻居embeddings
                    neighbor_embeddings_tensor = torch.stack(neighbor_embeddings)  # [num_neighbors, bert_dim]
                    e_context = torch.sum(
                        normalized_weights.unsqueeze(1) * neighbor_embeddings_tensor,
                        dim=0
                    )  # [bert_dim]

                    # 步骤5: 核心-上下文融合（公式4.9）
                    entity_embedding = alpha * e_core + (1 - alpha) * e_context

                batch_entity_embeddings.append(entity_embedding)

            # 将batch内的所有实体embedding堆叠
            batch_entity_embeddings_tensor = torch.stack(batch_entity_embeddings)  # [batch_size, bert_dim]
            all_entity_embeddings.append(batch_entity_embeddings_tensor.cpu())

            if (end_idx) % 1000 == 0:
                print(f"  已处理 {end_idx}/{self.num_entities} 个实体...")

        # 拼接所有batch
        entity_embeddings = torch.cat(all_entity_embeddings, dim=0)

        print(f"✓ 实体embedding生成完成: {entity_embeddings.shape}")

        # 关系embedding也使用置信度感知的上下文增强（基于包含该关系的三元组）
        print(f"正在生成关系embedding...")
        all_relation_embeddings = []

        for start_idx in range(0, self.num_relations, batch_size):
            end_idx = min(start_idx + batch_size, self.num_relations)
            relation_ids = list(range(start_idx, end_idx))

            batch_relation_embeddings = []

            for relation_id in relation_ids:
                rel_text = self.relation_id2text.get(relation_id, f"relation_{relation_id}")

                # 步骤1: 编码关系核心语义
                # 使用辅助词增强关系语义，如 "is the relation between A and B"
                core_text = f"{rel_text}"

                core_encoded = self.tokenizer(
                    core_text,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors='pt'
                )
                core_input_ids = core_encoded['input_ids'].to(self.device)
                core_attention_mask = core_encoded['attention_mask'].to(self.device)

                with torch.no_grad():
                    core_outputs = self.bert_model(input_ids=core_input_ids, attention_mask=core_attention_mask)
                    r_core = core_outputs.last_hidden_state[:, 0, :].squeeze(0)  # [bert_dim]

                # 步骤2: 获取包含该关系的三元组作为上下文
                # 从data对象获取包含该关系的三元组
                sampled_triplets = data.get_relation_triplets(relation_id, max_triplets=max_neighbors, min_weight=min_weight)

                # # 硬过滤：保留置信度 >= min_weight 的三元组
                # filtered_triplets = [t for t in relation_triplets if t[2] >= min_weight]

                # # 降序采样：按置信度排序，取top-K
                # filtered_triplets = sorted(filtered_triplets, key=lambda x: x[2], reverse=True)
                # sampled_triplets = filtered_triplets[:max_neighbors]

                if len(sampled_triplets) == 0:
                    # 没有三元组上下文，只使用核心语义
                    relation_embedding = r_core
                else:
                    # 步骤3: 编码三元组上下文（公式4.6）
                    context_embeddings = []
                    context_weights = []

                    for head_id, tail_id, weight in sampled_triplets:
                        head_text = self.entity_id2text.get(head_id, f"entity_{head_id}")
                        tail_text = self.entity_id2text.get(tail_id, f"entity_{tail_id}")

                        # 使用辅助词构建关系三元组文本（公式4.6的思想）
                        # "relation is the relation between head and tail"
                        triplet_text = f"{rel_text} is the relation between {head_text} and {tail_text}"

                        context_encoded = self.tokenizer(
                            triplet_text,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors='pt'
                        )
                        context_input_ids = context_encoded['input_ids'].to(self.device)
                        context_attention_mask = context_encoded['attention_mask'].to(self.device)

                        with torch.no_grad():
                            context_outputs = self.bert_model(
                                input_ids=context_input_ids,
                                attention_mask=context_attention_mask
                            )
                            r_context_i = context_outputs.last_hidden_state[:, 0, :].squeeze(0)

                        context_embeddings.append(r_context_i)
                        context_weights.append(weight)

                    # 步骤4: 置信度加权聚合（公式4.8）
                    context_weights_tensor = torch.tensor(context_weights, dtype=torch.float32, device=self.device)
                    normalized_weights = torch.nn.functional.softmax(context_weights_tensor / temperature, dim=0)

                    context_embeddings_tensor = torch.stack(context_embeddings)
                    r_context = torch.sum(
                        normalized_weights.unsqueeze(1) * context_embeddings_tensor,
                        dim=0
                    )

                    # 步骤5: 核心-上下文融合（公式4.10）
                    relation_embedding = alpha * r_core + (1 - alpha) * r_context

                batch_relation_embeddings.append(relation_embedding)

            # 堆叠batch内的所有关系embedding
            batch_relation_embeddings_tensor = torch.stack(batch_relation_embeddings)
            all_relation_embeddings.append(batch_relation_embeddings_tensor.cpu())

        relation_embeddings = torch.cat(all_relation_embeddings, dim=0)
        print(f"✓ 关系embedding生成完成: {relation_embeddings.shape}")
        print("=" * 70)

        return entity_embeddings, relation_embeddings

    def save_embeddings(self, entity_embeddings, relation_embeddings, save_dir):
        """
        保存BERT生成的embedding到文件

        Args:
            entity_embeddings: [num_entities, bert_dim]
            relation_embeddings: [num_relations, bert_dim]
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        entity_path = os.path.join(save_dir, 'bert_entity_embeddings.pt')
        relation_path = os.path.join(save_dir, 'bert_relation_embeddings.pt')

        torch.save(entity_embeddings, entity_path)
        torch.save(relation_embeddings, relation_path)

        print(f"✓ BERT embedding已保存:")
        print(f"  实体: {entity_path}")
        print(f"  关系: {relation_path}")

    def load_embeddings(self, save_dir):
        """
        从文件加载BERT embedding

        Args:
            save_dir: 保存目录

        Returns:
            entity_embeddings: [num_entities, bert_dim]
            relation_embeddings: [num_relations, bert_dim]
        """
        import os

        entity_path = os.path.join(save_dir, 'bert_entity_embeddings.pt')
        relation_path = os.path.join(save_dir, 'bert_relation_embeddings.pt')

        if not os.path.exists(entity_path) or not os.path.exists(relation_path):
            raise FileNotFoundError(f"BERT embedding文件不存在: {save_dir}")

        entity_embeddings = torch.load(entity_path)
        relation_embeddings = torch.load(relation_path)

        print(f"✓ BERT embedding已加载:")
        print(f"  实体: {entity_embeddings.shape}")
        print(f"  关系: {relation_embeddings.shape}")

        return entity_embeddings, relation_embeddings


def test_bert_encoder():
    """
    测试BERT编码器
    """
    print("=== 测试BERT编码器 ===")

    # 初始化编码器
    data_dir = './data/cn15k'
    encoder = BERTEncoder(data_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 测试1: 编码单个三元组
    print("\n测试1: 编码单个三元组")
    h_id = torch.tensor([0], dtype=torch.long).cuda()
    r_id = torch.tensor([0], dtype=torch.long).cuda()
    t_id = torch.tensor([1], dtype=torch.long).cuda()

    h_emb, r_emb, t_emb = encoder.encode_triplet_batch(h_id, r_id, t_id)
    print(f"Head embedding shape: {h_emb.shape}")
    print(f"Relation embedding shape: {r_emb.shape}")
    print(f"Tail embedding shape: {t_emb.shape}")

    # 测试2: 获取所有实体的embedding
    print("\n测试2: 获取前100个实体的embedding")
    # 只测试前100个实体
    encoder.num_entities = min(100, encoder.num_entities)
    all_entity_emb = encoder.get_all_entity_embeddings(batch_size=32)
    print(f"所有实体embedding shape: {all_entity_emb.shape}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_bert_encoder()
