from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import codecs
import numpy as np
import os
from src import utils
from src.bert_encoder import BERTEncoder
from src.cluster_embedding import ClusterEmbedding, GatedFusion, ResidualFusion, ClusterContrastiveLoss

class unKG(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):

        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, data, args, device, save_dir):
        super(unKG, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.data = data
        self.device = device
        self.args = args
        self.save_dir = save_dir

        # BERT编码器初始化（如果启用）
        self.use_bert = getattr(args, 'use_bert', False)
        self.bert_encoder = None

        # 聚类相关参数
        self.use_clustering = getattr(args, 'use_clustering', False)
        self.cluster_embedding = None
        self.entity_gate_fusion = None
        self.relation_gate_fusion = None
        self.cluster_contrastive_loss = None

        # 检查重新生成标志
        self.regenerate_bert = getattr(args, 'regenerate_bert', False)
        self.regenerate_clustering = getattr(args, 'regenerate_clustering', False)

        # 确定embedding缓存目录
        self.embedding_cache_dir = self.args.bert_cache_dir
        self.cluster_cache_dir = self.args.clustering_cache_dir

        # 线性投影层（如果需要）
        # BERT的768维 → 目标维度dim（可训练，参与梯度更新）
        self.entity_projection = None
        self.relation_projection = None

        # 初始化双分支MLP
        # 分支1: 置信度预测 - 输入concat(h,r,t) 3*dim
        # 分支2: 链接预测 - 输入concat(h,r,t,h+r-t) 4*dim
        self._init_dual_branch_mlp()

        # 初始化自适应加权损失的可学习不确定性参数
        self.var_conf = nn.Parameter(torch.ones(1, device=device, requires_grad=True))  # 置信度任务
        self.var_rank = nn.Parameter(torch.ones(1, device=device, requires_grad=True))  # 链接预测任务

        print("=" * 70)
        print("模型初始化流程")
        print("=" * 70)
        print(f"配置: use_bert={self.use_bert}, use_clustering={self.use_clustering}")
        print(f"重新生成标志: regenerate_bert={self.regenerate_bert}, regenerate_clustering={self.regenerate_clustering}")

        # 初始化BERT编码器（如果需要）
        if self.use_bert:
            self._init_bert_encoder()

        # 初始化embeddings（使用768维BERT维度，在forward中投影）
        # 注意：如果使用BERT，embedding_dim=768；否则使用self.dim
        self.bert_dim = 768 if self.use_bert else self.dim
        self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.bert_dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.bert_dim)

        # 根据不同场景初始化embeddings
        self._init_embeddings_conditional()

        # 如果使用聚类，初始化聚类相关模块
        if self.use_clustering:
            self._init_clustering_modules()

        print("=" * 70)
        print("模型初始化完成")
        print("=" * 70)

    def _init_dual_branch_mlp(self):
        """
        初始化双分支MLP架构（论文公式4.18-4.19）

        分支1: 置信度预测任务
            - 输入：concat(h, r, t) → 3*dim
            - 三层MLP
            - 输出：置信度分数 ∈ [0,1]

        分支2: 链接预测任务
            - 输入：concat(h, r, t, h+r-t) → 4*dim（结构化残差特征）
            - 三层MLP（参数独立）
            - 输出：排序分数 ∈ [0,1]
        """
        print("\n" + "-" * 70)
        print("初始化双分支MLP架构")
        print("-" * 70)

        # 输入：concat(h, r, t) = 3*dim
        self.confidence_mlp = nn.Sequential(
            nn.Linear(3 * self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim // 2, 1)
        ).to(self.device)

        # 分支2: 链接预测MLP（类似TransE的结构化残差）
        # 输入：concat(h, r, t, h+r-t) = 4*dim
        self.ranking_mlp = nn.Sequential(
            nn.Linear(4 * self.dim, 2 * self.dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim, 1)
        ).to(self.device)

        # 初始化权重
        for module in [self.confidence_mlp, self.ranking_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _init_bert_encoder(self):
        """初始化BERT编码器"""
        print("\n" + "-" * 70)
        print("初始化BERT编码器")
        print("-" * 70)

        bert_model_name = getattr(self.args, 'bert_model', 'bert-base-uncased')
        freeze_bert = getattr(self.args, 'freeze_bert', True)

        # 直接传递data对象，避免重复加载实体和关系映射
        self.bert_encoder = BERTEncoder(
            data_obj=self.data,
            model_name=bert_model_name,
            device=self.device,
            freeze_bert=freeze_bert
        )

        # BERT的输出维度（固定768）
        bert_dim = self.bert_encoder.bert_dim

        # 添加可训练的投影层：768维 → 目标维度
        # 这个投影层会参与训练，学习最优的降维映射
        # print(f"添加可训练投影层: BERT维度({bert_dim}) → 目标维度({self.dim})")
        self.entity_projection = nn.Linear(bert_dim, self.dim).to(self.device)
        self.relation_projection = nn.Linear(bert_dim, self.dim).to(self.device)

        # 初始化投影层权重
        nn.init.xavier_uniform_(self.entity_projection.weight)
        nn.init.zeros_(self.entity_projection.bias)
        nn.init.xavier_uniform_(self.relation_projection.weight)
        nn.init.zeros_(self.relation_projection.bias)

        print(f"✓ BERT编码器初始化完成")
        # print(f"✓ 投影层已创建（可训练参数）")
        print("-" * 70)

    def _init_embeddings_conditional(self):
        """
        根据不同场景条件初始化embeddings

        场景1: 不使用BERT和聚类 → 随机初始化
        场景2: 使用BERT，不重新生成，存在缓存 → 加载BERT初始化的embeddings
        场景3: 使用BERT，需要重新生成或无缓存 → 使用BERT重新初始化
        场景4: 使用聚类，不重新生成，存在缓存 → 加载训练过的embeddings（包括簇共享embeddings）
        """
        print("\n" + "-" * 70)
        print("条件初始化Embeddings")
        print("-" * 70)

        # 检查是否存在训练过的embeddings缓存
        embedding_cache_exists = self._check_embedding_cache_exists()

        if not self.use_bert and not self.use_clustering:
            # 场景1: 基线模型 - 随机初始化
            nn.init.xavier_uniform_(self.ent_embedding.weight)
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            print("  ✓ 随机初始化完成")

        elif self.use_bert and not self.regenerate_bert and embedding_cache_exists:
            # 场景2: 使用缓存的BERT初始化embeddings
            self._load_bert_initialized_embeddings()
            print("  ✓ BERT初始化embeddings加载完成")

        elif self.use_bert:
            # 场景3: 重新使用BERT初始化embeddings
            if self.regenerate_bert:
                print("场景: 重新生成BERT embeddings（--regenerate_bert已启用）")
            else:
                print("场景: BERT embeddings缓存不存在，开始生成")

            self._init_bert_embeddings()
            # 保存BERT初始化的embeddings
            self._save_bert_initialized_embeddings()
            print("  ✓ BERT embeddings已生成并保存")

        else:
            # 场景4: 使用聚类但不使用BERT - 随机初始化
            print("场景: 使用聚类但不使用BERT")
            print("  - 使用随机初始化embeddings")
            nn.init.xavier_uniform_(self.ent_embedding.weight)
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            print("  ✓ 随机初始化完成")

        print("-" * 70)

    def _check_embedding_cache_exists(self):
        """检查768维BERT embedding缓存是否存在"""
        entity_path = os.path.join(self.embedding_cache_dir, 'entity_bert_embeddings_768.pt')
        relation_path = os.path.join(self.embedding_cache_dir, 'relation_bert_embeddings_768.pt')
        return os.path.exists(entity_path) and os.path.exists(relation_path)

    def _save_bert_initialized_embeddings(self):
        """保存768维BERT embeddings（静态，用于后续直接加载）"""
        os.makedirs(self.embedding_cache_dir, exist_ok=True)

        entity_path = os.path.join(self.embedding_cache_dir, 'entity_bert_embeddings_768.pt')
        relation_path = os.path.join(self.embedding_cache_dir, 'relation_bert_embeddings_768.pt')

        # 保存768维原始BERT embeddings（不是投影后的）
        torch.save(self.ent_embedding.weight.data.cpu(), entity_path)
        torch.save(self.rel_embedding.weight.data.cpu(), relation_path)

        # print(f"  - 768维BERT实体embedding已保存: {entity_path}")
        # print(f"  - 768维BERT关系embedding已保存: {relation_path}")

    def _load_bert_initialized_embeddings(self):
        """加载768维BERT embeddings直接初始化embedding层（不投影）"""
        entity_path = os.path.join(self.embedding_cache_dir, 'entity_bert_embeddings_768.pt')
        relation_path = os.path.join(self.embedding_cache_dir, 'relation_bert_embeddings_768.pt')

        if not os.path.exists(entity_path) or not os.path.exists(relation_path):
            raise FileNotFoundError(f"Embedding缓存不存在: {self.embedding_cache_dir}")

        # 加载768维BERT embeddings
        entity_bert_emb = torch.load(entity_path).to(self.device)
        relation_bert_emb = torch.load(relation_path).to(self.device)

        # 直接用768维BERT初始化embedding层（可训练）
        with torch.no_grad():
            self.ent_embedding.weight.data.copy_(entity_bert_emb)
            self.rel_embedding.weight.data.copy_(relation_bert_emb)

        # print(f"  - 实体embedding(768维): {self.ent_embedding.weight.shape} (可训练)")
        # print(f"  - 关系embedding(768维): {self.rel_embedding.weight.shape} (可训练)")

    def _init_bert_embeddings(self):
        """
        使用BERT预训练的embedding初始化实体和关系embedding（上下文增强版本）
        直接用768维BERT初始化，投影在forward时动态进行
        """
        print("  - 正在使用BERT生成embeddings（上下文增强）...")

        # 获取上下文增强参数
        max_neighbors = getattr(self.args, 'max_neighbors', 5)
        min_weight = getattr(self.args, 'min_weight', 0.85)
        print(f"    上下文增强参数: max_neighbors={max_neighbors}, min_weight={min_weight}")

        # 获取所有实体和关系的上下文增强BERT embedding（768维）
        with torch.no_grad():
            entity_bert_emb, relation_bert_emb = self.bert_encoder.get_context_enhanced_embeddings(
                data=self.data,
                batch_size=32,
                max_neighbors=max_neighbors,
                min_weight=min_weight
            )

            # 移动到设备
            entity_bert_emb = entity_bert_emb.to(self.device)
            relation_bert_emb = relation_bert_emb.to(self.device)

        # 直接用768维BERT初始化embedding层（可训练）
        with torch.no_grad():
            self.ent_embedding.weight.data.copy_(entity_bert_emb)
            self.rel_embedding.weight.data.copy_(relation_bert_emb)

        # print(f"  - 实体embedding(768维): {self.ent_embedding.weight.shape} (可训练)")
        # print(f"  - 关系embedding(768维): {self.rel_embedding.weight.shape} (可训练)")

    def _init_clustering_modules(self):
        """
        初始化聚类相关模块
        使用768维BERT embeddings进行聚类，簇embedding也是768维可训练
        """
        print("\n" + "-" * 70)
        print("初始化聚类模块")
        print("-" * 70)

        # 导入聚类模块
        from src.clustering import perform_clustering

        # 使用768维个体embeddings进行聚类
        # 注意：这些embeddings已经在 _init_embeddings_conditional() 中用BERT初始化了
        entity_embeddings_768 = self.ent_embedding.weight.data.cpu()
        relation_embeddings_768 = self.rel_embedding.weight.data.cpu()

        print(f"  - 使用768维个体embeddings进行聚类")
        # print(f"    实体embeddings: {entity_embeddings_768.shape}")
        # print(f"    关系embeddings: {relation_embeddings_768.shape}")

        # 执行聚类或加载聚类缓存
        num_entity_clusters = self.args.num_entity_clusters
        num_relation_clusters = self.args.num_relation_clusters

        # 如果BERT重新生成了，聚类也需要重新执行
        need_reclustering = self.regenerate_clustering or self.regenerate_bert

        clustering = perform_clustering(
            entity_embeddings=entity_embeddings_768,
            relation_embeddings=relation_embeddings_768,
            cluster_cache_dir=self.cluster_cache_dir,
            num_entity_clusters=num_entity_clusters,
            num_relation_clusters=num_relation_clusters,
            regenerate_clustering=need_reclustering
        )

        # 检查是否存在训练过的簇共享embeddings
        cluster_emb_cache_exists = self._check_cluster_embedding_cache_exists()

        # 初始化簇共享Embedding模块（768维，可训练）
        self.cluster_embedding = ClusterEmbedding(
            num_entity_clusters=clustering.num_entity_clusters,
            num_relation_clusters=clustering.num_relation_clusters,
            embedding_dim=768,  # 768维
            entity_cluster_labels=clustering.entity_cluster_labels,
            relation_cluster_labels=clustering.relation_cluster_labels,
            entity_cluster_centers=clustering.entity_cluster_centers,  # 768维簇中心
            relation_cluster_centers=clustering.relation_cluster_centers,  # 768维簇中心
            device=self.device
        ).to(self.device)

        # 如果不重新聚类且缓存存在，加载训练过的簇共享embeddings
        if not self.regenerate_clustering and not self.regenerate_bert and cluster_emb_cache_exists:
            print("  - 检测到训练过的簇共享embeddings缓存")
            self._load_cluster_embeddings()
            print("  ✓ 簇共享embeddings已从缓存加载")
        else:
            if self.regenerate_clustering:
                print("  - 簇共享embeddings将从聚类中心重新初始化（--regenerate_clustering已启用）")
            elif self.regenerate_bert:
                print("  - 簇共享embeddings将从聚类中心重新初始化（BERT已重新生成）")
            else:
                print("  - 簇共享embeddings将从聚类中心初始化（首次运行）")

        # 初始化融合模块（残差融合）
        # 注意：融合模块的embedding_dim是目标维度self.dim
        self.entity_gate_fusion = ResidualFusion(
            embedding_dim=self.dim
        ).to(self.device)

        self.relation_gate_fusion = ResidualFusion(
            embedding_dim=self.dim
        ).to(self.device)

        # 初始化对比学习损失
        temperature = getattr(self.args, 'contrastive_temperature', 0.1)
        entity_weight = getattr(self.args, 'entity_contrastive_weight', 1.0)
        relation_weight = getattr(self.args, 'relation_contrastive_weight', 1.0)

        self.cluster_contrastive_loss = ClusterContrastiveLoss(
            temperature=temperature,
            entity_weight=entity_weight,
            relation_weight=relation_weight
        ).to(self.device)

        print(f"  - 对比学习温度: {temperature}")
        print(f"  - 实体对比学习权重: {entity_weight}")
        print(f"  - 关系对比学习权重: {relation_weight}")
        print(f"  - 投影层共享: entity_projection, relation_projection")
        print("-" * 70)
        print("✓ 聚类模块初始化完成")
        print("-" * 70)

    def _check_cluster_embedding_cache_exists(self):
        """检查训练过的簇共享embeddings缓存是否存在"""
        entity_cluster_path = os.path.join(self.cluster_cache_dir, 'entity_cluster_embeddings.pt')
        relation_cluster_path = os.path.join(self.cluster_cache_dir, 'relation_cluster_embeddings.pt')
        return os.path.exists(entity_cluster_path) and os.path.exists(relation_cluster_path)

    def _load_cluster_embeddings(self):
        """加载训练过的簇共享embeddings"""
        entity_cluster_path = os.path.join(self.cluster_cache_dir, 'entity_cluster_embeddings.pt')
        relation_cluster_path = os.path.join(self.cluster_cache_dir, 'relation_cluster_embeddings.pt')

        if not os.path.exists(entity_cluster_path) or not os.path.exists(relation_cluster_path):
            raise FileNotFoundError(f"簇共享embedding缓存不存在: {self.cluster_cache_dir}")

        entity_cluster_emb = torch.load(entity_cluster_path).to(self.device)
        relation_cluster_emb = torch.load(relation_cluster_path).to(self.device)

        self.cluster_embedding.entity_cluster_embeddings.weight.data.copy_(entity_cluster_emb)
        self.cluster_embedding.relation_cluster_embeddings.weight.data.copy_(relation_cluster_emb)

        print(f"    - 实体簇共享embedding: {entity_cluster_emb.shape}")
        print(f"    - 关系簇共享embedding: {relation_cluster_emb.shape}")

    def save_trained_embeddings(self):
        """
        保存训练后的所有embeddings（供后续直接加载使用）

        保存内容：
        - 实体个体embeddings
        - 关系个体embeddings
        - 实体簇共享embeddings（如果使用聚类）
        - 关系簇共享embeddings（如果使用聚类）
        """
        os.makedirs(self.embedding_cache_dir, exist_ok=True)

        # 保存实体和关系embeddings
        entity_path = os.path.join(self.embedding_cache_dir, 'entity_embeddings.pt')
        relation_path = os.path.join(self.embedding_cache_dir, 'relation_embeddings.pt')

        torch.save(self.ent_embedding.weight.data.cpu(), entity_path)
        torch.save(self.rel_embedding.weight.data.cpu(), relation_path)

        print(f"✓ 实体embedding已保存: {entity_path}")
        print(f"✓ 关系embedding已保存: {relation_path}")

        # 如果使用聚类，保存簇共享embeddings
        if self.use_clustering and self.cluster_embedding is not None:
            os.makedirs(self.cluster_cache_dir, exist_ok=True)

            entity_cluster_path = os.path.join(self.cluster_cache_dir, 'entity_cluster_embeddings.pt')
            relation_cluster_path = os.path.join(self.cluster_cache_dir, 'relation_cluster_embeddings.pt')

            torch.save(self.cluster_embedding.entity_cluster_embeddings.weight.data.cpu(), entity_cluster_path)
            torch.save(self.cluster_embedding.relation_cluster_embeddings.weight.data.cpu(), relation_cluster_path)

            print(f"✓ 实体簇共享embedding已保存: {entity_cluster_path}")
            print(f"✓ 关系簇共享embedding已保存: {relation_cluster_path}")

        print(f"✓ 所有embeddings已保存")

    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):

        # 转换为tensor并移动到设备
        h = torch.tensor(h, dtype=torch.int64).to(self.device)
        r = torch.tensor(r, dtype=torch.int64).to(self.device)
        t = torch.tensor(t, dtype=torch.int64).to(self.device)
        w = torch.tensor(w, dtype=torch.float32).to(self.device)
        n_hn = torch.tensor(n_hn, dtype=torch.int64).to(self.device)
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).to(self.device)
        n_t = torch.tensor(n_t, dtype=torch.int64).to(self.device)
        n_h = torch.tensor(n_h, dtype=torch.int64).to(self.device)
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).to(self.device)
        n_tn = torch.tensor(n_tn, dtype=torch.int64).to(self.device)


        head_fused, rel_fused, tail_fused = self.cal_dual_scores(h, r, t)
        n_hn_fused, n_rel_hn_fused, n_t_fused = self.cal_dual_scores(n_hn, n_rel_hn, n_t)
        n_h_fused, n_rel_tn_fused, n_tn_fused = self.cal_dual_scores(n_h, n_rel_tn, n_tn)


        # 步骤4：双分支计算
        # 正样本
        confidence = self.cal_confidence(head_fused, rel_fused, tail_fused)
        ranking_score = self.cal_ranking_score(head_fused, rel_fused, tail_fused)

        # 负样本：置信度和排序分数（头实体替换）
        ranking_neg_h = self.cal_ranking_score(n_hn_fused, n_rel_hn_fused, n_t_fused)
        # 负样本：置信度和排序分数（尾实体替换）
        ranking_neg_t = self.cal_ranking_score(n_h_fused, n_rel_tn_fused, n_tn_fused)

        # 损失1: 置信度预测损失（MSE）
        confidence_loss = self.confidence_prediction_loss(confidence, w)

        # 损失2: 置信度加权边际损失
        ranking_loss = self.confidence_weighted_margin_loss(
            ranking_score, ranking_neg_h, ranking_neg_t, w
        )

        # 损失3: L2正则化损失
        regularizer_loss = self.regularizer_loss(h, r, t, n_hn, n_tn)

        # 损失4: 对比学习损失（如果使用聚类）
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.use_clustering and self.cluster_contrastive_loss is not None:
            # 获取头实体和尾实体的簇标签
            head_cluster_labels = self.cluster_embedding.get_entity_cluster_id(h)
            tail_cluster_labels = self.cluster_embedding.get_entity_cluster_id(t)
            relation_cluster_labels = self.cluster_embedding.get_relation_cluster_id(r)

            # 拼接头尾实体及其簇标签（让头尾实体都参与实体级别的对比学习）
            entity_embeddings = torch.cat([head_fused, tail_fused], dim=0)  # [2*batch_size, dim]
            entity_cluster_labels = torch.cat([head_cluster_labels, tail_cluster_labels], dim=0)  # [2*batch_size]

            # 计算对比学习损失（直接使用已融合的embeddings，避免重复计算）
            contrastive_loss, entity_cl_loss, relation_cl_loss = self.cluster_contrastive_loss(
                entity_embeddings, entity_cluster_labels,  # 头尾实体一起参与
                rel_fused, relation_cluster_labels
            )

        # 自适应加权损失聚合
        total_loss = self.adaptive_weighted_loss(
            confidence_loss, ranking_loss, regularizer_loss, contrastive_loss * self.args.contrastive_weight
        )

        return total_loss, confidence_loss, ranking_loss, contrastive_loss * self.args.contrastive_weight, regularizer_loss * self.reg_scale

    def cal_dual_scores(self, h, r, t):
        """
        计算三元组的双分支分数：置信度分数 + 排序分数

        Args:
            h: 头实体ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            r: 关系ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            t: 尾实体ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            return_embeddings: 是否返回融合后的embeddings（用于对比学习，避免重复计算）

        Returns:
             head_fused, rel_fused, tail_fused
        """
        # 步骤1：获取768维个体embeddings（可训练）
        head_768 = self.ent_embedding(h)  # [batch_size, 768] 或 [batch_size, neg_num, 768]
        rel_768 = self.rel_embedding(r)
        tail_768 = self.ent_embedding(t)

        # 步骤2：投影到目标维度（投影层在每个batch都被使用，持续学习）
        if self.use_bert:
            head = self.entity_projection(head_768)  # [batch_size, dim]
            rel = self.relation_projection(rel_768)
            tail = self.entity_projection(tail_768)
        else:
            # 如果不使用BERT，embedding本身就是目标维度
            head = head_768
            rel = rel_768
            tail = tail_768

        # 步骤3：如果使用聚类，获取簇embeddings并融合
        if self.use_clustering and self.cluster_embedding is not None:
            # 获取768维簇embeddings（可训练）
            head_cluster_768 = self.cluster_embedding.get_entity_cluster_embedding(h)
            rel_cluster_768 = self.cluster_embedding.get_relation_cluster_embedding(r)
            tail_cluster_768 = self.cluster_embedding.get_entity_cluster_embedding(t)

            # 投影到目标维度（共享投影层）
            head_cluster = self.entity_projection(head_cluster_768)
            rel_cluster = self.relation_projection(rel_cluster_768)
            tail_cluster = self.entity_projection(tail_cluster_768)

            # 残差融合（在目标维度上融合）
            head_fused, _ = self.entity_gate_fusion(head, head_cluster)
            rel_fused, _ = self.relation_gate_fusion(rel, rel_cluster)
            tail_fused, _ = self.entity_gate_fusion(tail, tail_cluster)
        else:
            # 如果不使用聚类，直接使用投影后的个体embeddings
            head_fused = head
            rel_fused = rel
            tail_fused = tail


        return head_fused, rel_fused, tail_fused


    def cal_score(self, h, r, t, return_embeddings=False):
        """
        计算三元组的置信度分数

        Args:
            h: 头实体ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            r: 关系ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            t: 尾实体ID
                - 正样本: [batch_size]
                - 负样本: [batch_size, neg_num]
            return_embeddings: 是否返回融合后的embeddings（用于对比学习，避免重复计算）

        Returns:
            如果 return_embeddings=False:
                confidence: 置信度分数
                    - 正样本: [batch_size, 1]
                    - 负样本: [batch_size, neg_num, 1]
            如果 return_embeddings=True:
                confidence, head_fused, rel_fused, tail_fused
                    - confidence: 置信度分数
                    - head_fused: 融合后的头实体embedding [batch_size, dim]
                    - rel_fused: 融合后的关系embedding [batch_size, dim]
                    - tail_fused: 融合后的尾实体embedding [batch_size, dim]
        """
        # 步骤1：获取768维个体embeddings（可训练）
        head_768 = self.ent_embedding(h)
        rel_768 = self.rel_embedding(r)
        tail_768 = self.ent_embedding(t)

        # 步骤2：投影到目标维度
        if self.use_bert:
            head = self.entity_projection(head_768)
            rel = self.relation_projection(rel_768)
            tail = self.entity_projection(tail_768)
        else:
            head = head_768
            rel = rel_768
            tail = tail_768

        # 步骤3：如果使用聚类，获取簇embeddings并融合
        if self.use_clustering and self.cluster_embedding is not None:
            # 获取768维簇embeddings（可训练）
            head_cluster_768 = self.cluster_embedding.get_entity_cluster_embedding(h)
            rel_cluster_768 = self.cluster_embedding.get_relation_cluster_embedding(r)
            tail_cluster_768 = self.cluster_embedding.get_entity_cluster_embedding(t)

            # 投影到目标维度（共享投影层）
            head_cluster = self.entity_projection(head_cluster_768)
            rel_cluster = self.relation_projection(rel_cluster_768)
            tail_cluster = self.entity_projection(tail_cluster_768)

            # 残差融合
            head_fused, _ = self.entity_gate_fusion(head, head_cluster)
            rel_fused, _ = self.relation_gate_fusion(rel, rel_cluster)
            tail_fused, _ = self.entity_gate_fusion(tail, tail_cluster)
        else:
            # 如果不使用聚类，直接使用投影后的个体embeddings
            head_fused = head
            rel_fused = rel
            tail_fused = tail

        # 计算置信度
        confidence = self.cal_confidence(head_fused, rel_fused, tail_fused)

        # 根据参数决定返回内容
        if return_embeddings:
            return confidence, head_fused, rel_fused, tail_fused
        else:
            return confidence
    def cal_confidence(self, head, rel, tail):
        """
        使用置信度预测MLP计算置信度分数（论文公式4.18-4.19）

        流程：
        1. 拼接：concat(h, r, t) → 3*dim
        2. MLP：三层非线性映射 → 似然性得分
        3. 激活：sigmoid/clamp → 置信度 ∈ [0,1]

        Args:
            head: 头实体embedding
                - 正样本: [batch_size, dim]
                - 负样本: [batch_size, neg_num, dim]
            rel: 关系embedding
                - 正样本: [batch_size, dim]
                - 负样本: [batch_size, neg_num, dim]
            tail: 尾实体embedding
                - 正样本: [batch_size, dim]
                - 负样本: [batch_size, neg_num, dim]

        Returns:
            confidence: 置信度分数 ∈ [0,1]
                - 正样本: [batch_size, 1]
                - 负样本: [batch_size, neg_num, 1]
        """
        # 步骤1: 拼接 concat(h, r, t) → 3*dim
        # 自动适配正负样本的不同维度
        concat_features = torch.cat([head, rel, tail], dim=-1)
        # 正样本: [batch_size, 3*dim]
        # 负样本: [batch_size, neg_num, 3*dim]

        # 步骤2: 通过置信度预测MLP（三层）
        likelihood_score = self.confidence_mlp(concat_features)
        # 输出: [batch_size, 1] 或 [batch_size, neg_num, 1]

        # 步骤3: 转换为置信度（使用逻辑函数）

        confidence = torch.sigmoid(likelihood_score)  # sigmoid激活


        return confidence  # [batch_size, 1] 或 [batch_size, neg_num, 1]

    def cal_ranking_score(self, head, rel, tail):
        """
        使用链接预测MLP计算排序分数（用于链接预测任务）

        流程：
        1. 拼接：concat(h, r, t, h+r-t) → 4*dim（类似TransE的结构化残差）
        2. MLP：三层非线性映射 → 排序得分
        3. 激活：sigmoid → 排序分数 ∈ [0,1]

        Args:
            head: 头实体embedding [batch_size, dim] 或 [batch_size, neg_num, dim]
            rel: 关系embedding [batch_size, dim] 或 [batch_size, neg_num, dim]
            tail: 尾实体embedding [batch_size, dim] 或 [batch_size, neg_num, dim]

        Returns:
            ranking_score: 排序分数 ∈ [0,1]
                - [batch_size, 1] 或 [batch_size, neg_num, 1]
        """
        # 步骤1: 计算结构化残差特征 h+r-t（TransE风格）
        structural_residual = head + rel - tail  # [batch_size, dim] 或 [batch_size, neg_num, dim]

        # 步骤2: 拼接 concat(h, r, t, h+r-t) → 4*dim
        concat_features = torch.cat([head, rel, tail, structural_residual], dim=-1)
        # 正样本: [batch_size, 4*dim]
        # 负样本: [batch_size, neg_num, 4*dim]

        # 步骤3: 通过链接预测MLP（三层）
        ranking_score = self.ranking_mlp(concat_features)
        # 输出: [batch_size, 1] 或 [batch_size, neg_num, 1]

        # 步骤4: sigmoid归一化
        ranking_score = torch.sigmoid(ranking_score)

        return ranking_score  # [batch_size, 1] 或 [batch_size, neg_num, 1]


    def confidence_prediction_loss(self, hrt, w):
        """
        置信度预测损失：MSE损失，专注于准确预测置信度值

        Args:
            hrt: 正例预测分数 (batch_size, 1)
            w: 正例实际置信度 (batch_size,) - 1维tensor
        Returns:
            confidence_loss: 置信度预测损失（标量）
        """
        w = torch.unsqueeze(w, dim=-1)  # [batch_size,] → [batch_size, 1]

        # 正例的置信度预测损失（MSE）
        confidence_loss = torch.square(hrt - w)  # [batch_size, 1]

        # 对batch求平均，返回标量
        return confidence_loss.mean()  # 标量

    def regularizer_loss(self, h, r, t, n_hn, n_tn):
        """
        L2正则化损失：防止embedding过拟合

        对正样本和负样本的所有embeddings进行L2正则化

        Args:
            h: 正样本头实体ID [batch_size]
            r: 正样本关系ID [batch_size]
            t: 正样本尾实体ID [batch_size]
            n_hn: 负样本头实体ID [batch_size, neg_num]
            n_tn: 负样本尾实体ID [batch_size, neg_num]

        Returns:
            regularizer_loss: L2正则化损失（标量）
        """
        # 获取正样本embeddings
        head = self.ent_embedding(h)  # [batch_size, dim]
        rel = self.rel_embedding(r)  # [batch_size, dim]
        tail = self.ent_embedding(t)  # [batch_size, dim]

        # 获取负样本embeddings
        neg_head = self.ent_embedding(n_hn)  # [batch_size, neg_num, dim]
        neg_tail = self.ent_embedding(n_tn)  # [batch_size, neg_num, dim]

        # 计算L2范数（正样本）
        pos_reg = (torch.mean(torch.square(head)) +
                   torch.mean(torch.square(rel)) +
                   torch.mean(torch.square(tail))) / 3.0

        # 计算L2范数（负样本）
        neg_reg = (torch.mean(torch.square(neg_head)) +
                   torch.mean(torch.square(neg_tail))) / 2.0

        # 总正则化损失（正负样本平均）
        regularizer_loss = (pos_reg + neg_reg) / 2.0

        return regularizer_loss

    def confidence_weighted_margin_loss(self, ranking_pos, ranking_neg_h, ranking_neg_t, w, margin=0.1):
        """
        置信度加权边际损失

        为了使模型在处理不确定事实时具有更强的鲁棒性，引入基于置信度加权的边际损失。
        越可信的事实（w越大），模型越强制要求其与负样本拉开距离；
        而对于低置信度的模糊事实，模型允许较小的间距，以防止噪声过拟合。

        Args:
            ranking_pos: 正样本排序分数 [batch_size, 1]
            ranking_neg_h: 负样本头部替换的排序分数 [batch_size, neg_num, 1]
            ranking_neg_t: 负样本尾部替换的排序分数 [batch_size, neg_num, 1]
            w: 正样本的真实置信度 [batch_size,]
            margin: 预设的间隔常数（默认1.0）

        Returns:
            ranking_loss: 置信度加权边际损失（标量）
        """
        w = torch.unsqueeze(w, dim=-1)  # [batch_size,] → [batch_size, 1]

        # 计算头实体替换的边际损失
        # w * max(0, margin - (ranking_pos - ranking_neg_h))
        # ranking_pos: [batch_size, 1]
        # ranking_neg_h: [batch_size, neg_num, 1]
        # 广播后 ranking_pos - ranking_neg_h: [batch_size, neg_num, 1]
        margin_h = margin - (ranking_pos.unsqueeze(1) - ranking_neg_h)  # [batch_size, neg_num, 1]
        loss_h = w.unsqueeze(1) * torch.relu(margin_h)  # [batch_size, neg_num, 1]

        # 先对neg_num维度求平均，再对batch维度求平均
        # 这样每个正样本的贡献权重相同，不受neg_num影响
        loss_h = loss_h.mean(dim=1)  # [batch_size, 1] - 对每个样本的所有负样本求平均
        loss_h = loss_h.mean()  # 标量 - 对batch求平均

        # 计算尾实体替换的边际损失
        margin_t = margin - (ranking_pos.unsqueeze(1) - ranking_neg_t)  # [batch_size, neg_num, 1]
        loss_t = w.unsqueeze(1) * torch.relu(margin_t)  # [batch_size, neg_num, 1]

        # 先对neg_num维度求平均，再对batch维度求平均
        loss_t = loss_t.mean(dim=1)  # [batch_size, 1]
        loss_t = loss_t.mean()  # 标量

        # 总边际损失
        ranking_loss = (loss_h + loss_t) / 2.0

        return ranking_loss

    def adaptive_weighted_loss(self, confidence_loss, ranking_loss, regularizer_loss, contrastive_loss):
        """
        自适应加权损失函数（基于Alex Kendall, CVPR 2018）

        基于同方差不确定性（Homoscedastic Uncertainty）的自适应加权机制。
        通过为每个任务引入可学习的观测噪声参数σ²，模型可以自动预测各任务的相对噪声水平，
        并在训练过程中动态赋予损失函数不同的权重。

        参考文献：
        Kendall, Alex, et al. "Multi-task learning using uncertainty to weigh losses
        for scene geometry and semantics." CVPR 2018.

        Auxiliary Tasks in Multi-task Learning

        任务类型：
        - confidence_loss (MSE): 回归任务
        - ranking_loss (Margin): 分类任务


        Args:
            confidence_loss: 置信度预测损失（标量，MSE回归任务）
            ranking_loss: 链接预测损失（标量，Margin分类任务）
            regularizer_loss: L2正则化损失（标量）
            contrastive_loss: 对比学习损失（标量，InfoNCE分类任务）

        Returns:
            total_loss: 自适应加权后的总损失（标量）
        """

        # 自适应加权的三个主要任务损失（按照Kendall 2018公式）

        # 1. 置信度预测（回归任务）
        weighted_conf_loss = (0.5 / (self.var_conf ** 2)) * confidence_loss + torch.log(1 + self.var_conf ** 2)

        # 2. 链接预测（分类任务）
        weighted_rank_loss = (0.5 / (self.var_rank ** 2)) * ranking_loss + torch.log(1 + self.var_rank ** 2)

        # 3. 对比学习（分类任务）（不参与自适应加权）
        weighted_contrast_loss = contrastive_loss

        weighted_reg_loss = self.reg_scale * regularizer_loss

        # 总损失
        total_loss = weighted_conf_loss + weighted_rank_loss + weighted_contrast_loss + weighted_reg_loss

        return total_loss

