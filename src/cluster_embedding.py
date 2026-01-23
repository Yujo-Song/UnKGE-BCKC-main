"""
Cluster-level Shared Embeddings Module
簇级别共享Embedding模块

为每个实体簇和关系簇维护可训练的共享embedding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class ClusterEmbedding(nn.Module):
    """
    簇共享Embedding模块

    功能：
    1. 为每个实体簇维护一个可训练的共享embedding
    2. 为每个关系簇维护一个可训练的共享embedding
    3. 根据实体/关系ID，返回对应的簇共享embedding
    """

    def __init__(self, num_entity_clusters, num_relation_clusters,
                 embedding_dim, entity_cluster_labels, relation_cluster_labels,
                 entity_cluster_centers=None, relation_cluster_centers=None,
                 device='cuda'):
        """
        初始化簇共享Embedding模块

        Args:
            num_entity_clusters: 实体簇数量
            num_relation_clusters: 关系簇数量
            embedding_dim: embedding维度
            entity_cluster_labels: [num_entities] 每个实体的簇标签
            relation_cluster_labels: [num_relations] 每个关系的簇标签
            entity_cluster_centers: [num_entity_clusters, embedding_dim] 实体簇中心（用于初始化）
            relation_cluster_centers: [num_relation_clusters, embedding_dim] 关系簇中心（用于初始化）
            device: 计算设备
        """
        super(ClusterEmbedding, self).__init__()

        self.num_entity_clusters = num_entity_clusters
        self.num_relation_clusters = num_relation_clusters
        self.embedding_dim = embedding_dim
        self.device = device

        # 保存簇标签映射 (numpy数组，不需要梯度)
        self.entity_cluster_labels = torch.LongTensor(entity_cluster_labels).to(device)
        self.relation_cluster_labels = torch.LongTensor(relation_cluster_labels).to(device)

        # 创建可训练的簇共享embeddings
        self.entity_cluster_embeddings = nn.Embedding(
            num_embeddings=num_entity_clusters,
            embedding_dim=embedding_dim
        )

        self.relation_cluster_embeddings = nn.Embedding(
            num_embeddings=num_relation_clusters,
            embedding_dim=embedding_dim
        )

        # 初始化簇embeddings
        self._initialize_cluster_embeddings(
            entity_cluster_centers,
            relation_cluster_centers
        )

        print(f"簇共享Embedding模块初始化完成:")
        # print(f"  - 实体簇数量: {num_entity_clusters}")
        # print(f"  - 关系簇数量: {num_relation_clusters}")
        # print(f"  - Embedding维度: {embedding_dim}")
        # print(f"  - 设备: {device}")

    def _initialize_cluster_embeddings(self, entity_cluster_centers, relation_cluster_centers):
        """
        初始化簇embeddings

        如果提供了簇中心，使用簇中心初始化；否则使用xavier初始化
        """
        if entity_cluster_centers is not None:
            # 使用聚类得到的簇中心初始化
            print("  使用聚类簇中心初始化实体簇embeddings")
            if isinstance(entity_cluster_centers, np.ndarray):
                entity_cluster_centers = torch.FloatTensor(entity_cluster_centers)
            self.entity_cluster_embeddings.weight.data.copy_(entity_cluster_centers)
        else:
            # 使用Xavier初始化
            print("  使用Xavier初始化实体簇embeddings")
            nn.init.xavier_uniform_(self.entity_cluster_embeddings.weight)

        if relation_cluster_centers is not None:
            # 使用聚类得到的簇中心初始化
            print("  使用聚类簇中心初始化关系簇embeddings")
            if isinstance(relation_cluster_centers, np.ndarray):
                relation_cluster_centers = torch.FloatTensor(relation_cluster_centers)
            self.relation_cluster_embeddings.weight.data.copy_(relation_cluster_centers)
        else:
            # 使用Xavier初始化
            print("  使用Xavier初始化关系簇embeddings")
            nn.init.xavier_uniform_(self.relation_cluster_embeddings.weight)

    def get_entity_cluster_embedding(self, entity_ids):
        """
        获取实体的簇共享embedding

        Args:
            entity_ids: 实体ID张量 [batch_size] 或 [batch_size, neg_num]

        Returns:
            cluster_embeddings: 簇共享embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]
        """
        # 获取实体对应的簇ID
        cluster_ids = self.entity_cluster_labels[entity_ids]

        # 查询簇embeddings
        cluster_embeddings = self.entity_cluster_embeddings(cluster_ids)

        return cluster_embeddings

    def get_relation_cluster_embedding(self, relation_ids):
        """
        获取关系的簇共享embedding

        Args:
            relation_ids: 关系ID张量 [batch_size] 或 [batch_size, neg_num]

        Returns:
            cluster_embeddings: 簇共享embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]
        """
        # 获取关系对应的簇ID
        cluster_ids = self.relation_cluster_labels[relation_ids]

        # 查询簇embeddings
        cluster_embeddings = self.relation_cluster_embeddings(cluster_ids)

        return cluster_embeddings

    def get_entity_cluster_id(self, entity_ids):
        """
        获取实体的簇ID

        Args:
            entity_ids: 实体ID张量

        Returns:
            cluster_ids: 簇ID张量
        """
        return self.entity_cluster_labels[entity_ids]

    def get_relation_cluster_id(self, relation_ids):
        """
        获取关系的簇ID

        Args:
            relation_ids: 关系ID张量

        Returns:
            cluster_ids: 簇ID张量
        """
        return self.relation_cluster_labels[relation_ids]

    def forward(self, entity_ids=None, relation_ids=None):
        """
        前向传播

        Args:
            entity_ids: 实体ID张量（可选）
            relation_ids: 关系ID张量（可选）

        Returns:
            entity_cluster_emb: 实体簇embedding（如果提供了entity_ids）
            relation_cluster_emb: 关系簇embedding（如果提供了relation_ids）
        """
        results = []

        if entity_ids is not None:
            entity_cluster_emb = self.get_entity_cluster_embedding(entity_ids)
            results.append(entity_cluster_emb)

        if relation_ids is not None:
            relation_cluster_emb = self.get_relation_cluster_embedding(relation_ids)
            results.append(relation_cluster_emb)

        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            return results
        else:
            return None


class GatedFusion(nn.Module):
    """
    门控融合机制

    将个体embedding和簇共享embedding通过门控机制融合
    最终embedding = gate * individual_embedding + (1 - gate) * cluster_embedding
    """

    def __init__(self, embedding_dim, fusion_type='concat'):
        """
        初始化门控融合模块

        Args:
            embedding_dim: embedding维度
            fusion_type: 融合类型
                - 'concat': 拼接两个embedding，通过MLP计算gate
                - 'simple': 简单线性变换计算gate
        """
        super(GatedFusion, self).__init__()

        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 拼接方式：输入为2*embedding_dim，输出为1（gate值）
            self.gate_network = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()  # gate值在[0, 1]之间
            )
        elif fusion_type == 'simple':
            # 简单方式：直接从个体embedding计算gate
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

        # 初始化权重
        self._init_weights()

        print(f"门控融合模块初始化完成:")
        print(f"  - Embedding维度: {embedding_dim}")
        print(f"  - 融合类型: {fusion_type}")

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, individual_embedding, cluster_embedding):
        """
        前向传播：融合个体embedding和簇embedding

        Args:
            individual_embedding: 个体embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]
            cluster_embedding: 簇共享embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]

        Returns:
            fused_embedding: 融合后的embedding，形状与输入相同
            gate: 门控值，用于可解释性分析
        """
        if self.fusion_type == 'concat':
            # 拼接两个embedding
            concatenated = torch.cat([individual_embedding, cluster_embedding], dim=-1)
            # 计算gate值
            gate = self.gate_network(concatenated)  # [batch_size, 1] 或 [batch_size, neg_num, 1]
        else:  # simple
            # 仅使用个体embedding计算gate
            gate = self.gate_network(individual_embedding)  # [batch_size, 1] 或 [batch_size, neg_num, 1]

        # 门控融合
        fused_embedding = gate * individual_embedding + (1 - gate) * cluster_embedding

        return fused_embedding, gate


class ResidualFusion(nn.Module):
    """
    残差融合机制

    核心思想：将簇embedding作为对个体embedding的残差修正
    fused = individual + gate * transform(cluster - individual)

    优点：
    1. 保留个体信息为主体，簇信息作为修正
    2. 训练稳定，梯度流畅（残差连接）
    3. 门控控制修正强度，自适应调整
    """

    def __init__(self, embedding_dim, fusion_type='concat'):
        """
        初始化残差融合模块

        Args:
            embedding_dim: embedding维度
            fusion_type: 门控计算方式
                - 'concat': 拼接两个embedding计算gate
                - 'simple': 仅使用个体embedding计算gate
        """
        super(ResidualFusion, self).__init__()

        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type

        # 残差变换网络（将簇-个体的差异进行非线性变换）
        self.residual_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 门控网络（控制接受多少残差修正）
        if fusion_type == 'concat':
            # 基于个体和簇的拼接计算gate
            self.residual_gate = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        elif fusion_type == 'simple':
            # 仅基于个体embedding计算gate
            self.residual_gate = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

        # 初始化权重
        self._init_weights()

        # print(f"残差融合模块初始化完成:")
        # print(f"  - Embedding维度: {embedding_dim}")
        # print(f"  - 融合类型: {fusion_type}")
        # print(f"  - 残差变换: 2层MLP + LayerNorm + Dropout")

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, individual_embedding, cluster_embedding):
        """
        前向传播：残差融合个体embedding和簇embedding

        计算流程：
        1. 计算残差：residual = cluster - individual
        2. 变换残差：transformed_residual = MLP(residual)
        3. 计算门控：gate = sigmoid(MLP(individual, cluster))
        4. 残差融合：fused = individual + gate * transformed_residual

        Args:
            individual_embedding: 个体embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]
            cluster_embedding: 簇共享embedding [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]

        Returns:
            fused_embedding: 融合后的embedding，形状与输入相同
            gate: 门控值 [batch_size, 1] 或 [batch_size, neg_num, 1]，表示残差修正的强度
        """
        # 步骤1：计算残差（簇与个体的差异）
        residual = cluster_embedding - individual_embedding
        # [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]

        # 步骤2：对残差进行非线性变换
        transformed_residual = self.residual_transform(residual)
        # [batch_size, embedding_dim] 或 [batch_size, neg_num, embedding_dim]

        # 步骤3：计算门控值（决定接受多少残差修正）
        if self.fusion_type == 'concat':
            gate_input = torch.cat([individual_embedding, cluster_embedding], dim=-1)
            gate = self.residual_gate(gate_input)
        else:  # simple
            gate = self.residual_gate(individual_embedding)
        # [batch_size, 1] 或 [batch_size, neg_num, 1]

        # 步骤4：残差融合（保持个体为主，添加门控的残差修正）
        fused_embedding = individual_embedding + gate * transformed_residual

        return fused_embedding, gate


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数

    目标：
    1. 同一簇内的实体/关系embeddings应该相似（拉近）
    2. 不同簇之间的实体/关系embeddings应该不同（推远）

    使用InfoNCE损失实现
    """

    def __init__(self, temperature=0.1):
        """
        初始化对比学习损失

        Args:
            temperature: 温度参数，控制相似度的平滑程度
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, cluster_labels):
        """
        计算对比学习损失（InfoNCE）

        Args:
            embeddings: [batch_size, embedding_dim] 实体或关系的embeddings
            cluster_labels: [batch_size] 每个embedding对应的簇标签

        Returns:
            loss: 对比学习损失标量
        """
        batch_size = embeddings.shape[0]

        # 归一化embeddings（用于计算余弦相似度）
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)

        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature

        # 创建标签矩阵：相同簇为1，不同簇为0
        # [batch_size, 1] == [1, batch_size] -> [batch_size, batch_size]
        labels_matrix = (cluster_labels.unsqueeze(1) == cluster_labels.unsqueeze(0)).float()

        # 移除对角线（自己和自己的相似度）
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        labels_matrix = labels_matrix.masked_fill(mask, 0)

        # InfoNCE损失
        # 对于每个样本，正样本是同簇的其他样本，负样本是不同簇的样本
        # 使用数值稳定的logsumexp计算

        # 计算每行的正样本数量
        positive_counts = labels_matrix.sum(dim=1)  # [batch_size]

        # 只对有正样本的行计算损失
        valid_samples = positive_counts > 0

        if valid_samples.sum() == 0:
            # 如果batch中没有任何正样本对，返回0损失
            return torch.tensor(0.0, device=embeddings.device)

        # 计算分子：正样本对的相似度和
        positive_similarity = (similarity_matrix * labels_matrix).sum(dim=1)  # [batch_size]

        # 计算分母：所有样本对的相似度（logsumexp）
        # 排除自己
        exp_similarity = torch.exp(similarity_matrix).masked_fill(mask, 0)
        denominator = torch.log(exp_similarity.sum(dim=1) + 1e-8)  # [batch_size]

        # 对于每个有效样本，计算损失
        # loss = -log(exp(sim_pos) / sum(exp(sim_all)))
        # = -sim_pos + log(sum(exp(sim_all)))
        loss_per_sample = -positive_similarity / (positive_counts + 1e-8) + denominator

        # 只对有效样本求平均
        loss = loss_per_sample[valid_samples].mean()

        return loss


class ClusterContrastiveLoss(nn.Module):
    """
    簇级别对比学习损失

    结合实体和关系的对比学习损失
    """

    def __init__(self, temperature=0.1, entity_weight=1.0, relation_weight=1.0):
        """
        初始化簇级别对比学习损失

        Args:
            temperature: 温度参数
            entity_weight: 实体对比损失的权重
            relation_weight: 关系对比损失的权重
        """
        super(ClusterContrastiveLoss, self).__init__()

        self.entity_contrastive = ContrastiveLoss(temperature)
        self.relation_contrastive = ContrastiveLoss(temperature)

        self.entity_weight = entity_weight
        self.relation_weight = relation_weight

        # print(f"簇级别对比学习损失初始化完成:")
        # print(f"  - 实体损失权重: {entity_weight}")
        # print(f"  - 关系损失权重: {relation_weight}")

    def forward(self, entity_embeddings, entity_cluster_labels,
                relation_embeddings, relation_cluster_labels):
        """
        计算总对比学习损失

        Args:
            entity_embeddings: [batch_size, embedding_dim] 实体embeddings
            entity_cluster_labels: [batch_size] 实体簇标签
            relation_embeddings: [batch_size, embedding_dim] 关系embeddings
            relation_cluster_labels: [batch_size] 关系簇标签

        Returns:
            total_loss: 总对比学习损失
            entity_loss: 实体对比损失（用于日志）
            relation_loss: 关系对比损失（用于日志）
        """
        # 计算实体对比损失
        entity_loss = self.entity_contrastive(entity_embeddings, entity_cluster_labels)

        # 计算关系对比损失
        relation_loss = self.relation_contrastive(relation_embeddings, relation_cluster_labels)

        # 加权求和
        total_loss = self.entity_weight * entity_loss + self.relation_weight * relation_loss

        return total_loss, entity_loss, relation_loss
