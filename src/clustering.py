"""
Static Hierarchical Clustering for Knowledge Graph Entities and Relations
使用BERT embeddings进行静态层次聚类
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import os
from os.path import join
from sklearn.cluster import AgglomerativeClustering
import pickle


class KGClustering:
    """
    知识图谱聚类器

    功能：
    1. 对实体和关系的BERT embeddings进行层次聚类
    2. 保存和加载聚类结果
    3. 初始化簇中心embeddings
    """

    def __init__(self, num_entity_clusters, num_relation_clusters, embedding_dim):
        """
        初始化聚类器

        Args:
            num_entity_clusters: 实体簇数量
            num_relation_clusters: 关系簇数量
            embedding_dim: embedding维度
        """
        self.num_entity_clusters = num_entity_clusters
        self.num_relation_clusters = num_relation_clusters
        self.embedding_dim = embedding_dim

        # 聚类标签
        self.entity_cluster_labels = None  # [num_entities]
        self.relation_cluster_labels = None  # [num_relations]

        # 簇中心embeddings
        self.entity_cluster_centers = None  # [num_entity_clusters, embedding_dim]
        self.relation_cluster_centers = None  # [num_relation_clusters, embedding_dim]

        print(f"聚类器初始化完成:")
        print(f"  - 实体簇数量: {num_entity_clusters}")
        print(f"  - 关系簇数量: {num_relation_clusters}")
        print(f"  - Embedding维度: {embedding_dim}")

    def cluster_entities(self, entity_embeddings, method='ward', metric='euclidean'):
        """
        对实体embeddings进行层次聚类

        Args:
            entity_embeddings: [num_entities, embedding_dim] BERT生成的实体embeddings
            method: 聚类连接方法，默认'ward'（最小方差）
            metric: 距离度量，默认'euclidean'

        Returns:
            entity_cluster_labels: [num_entities] 每个实体的簇标签
            entity_cluster_centers: [num_entity_clusters, embedding_dim] 簇中心embeddings
        """
        print("=" * 70)
        print("正在对实体进行层次聚类...")
        print(f"  - 实体数量: {entity_embeddings.shape[0]}")
        print(f"  - 目标簇数: {self.num_entity_clusters}")
        print(f"  - 聚类方法: {method}")
        print(f"  - 距离度量: {metric}")

        # 转换为numpy数组（如果是tensor）
        if isinstance(entity_embeddings, torch.Tensor):
            entity_embeddings_np = entity_embeddings.cpu().numpy()
        else:
            entity_embeddings_np = entity_embeddings

        # 层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=self.num_entity_clusters,
            linkage=method,
            metric=metric
        )

        self.entity_cluster_labels = clustering.fit_predict(entity_embeddings_np)

        # 计算簇中心（每个簇内实体embeddings的平均值）
        self.entity_cluster_centers = self._compute_cluster_centers(
            entity_embeddings_np,
            self.entity_cluster_labels,
            self.num_entity_clusters
        )

        # 统计每个簇的大小
        cluster_sizes = np.bincount(self.entity_cluster_labels)
        print(f"✓ 实体聚类完成")
        print(f"  - 簇大小统计: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
              f"mean={cluster_sizes.mean():.2f}, std={cluster_sizes.std():.2f}")
        print("=" * 70)

        return self.entity_cluster_labels, self.entity_cluster_centers

    def cluster_relations(self, relation_embeddings, method='ward', metric='euclidean'):
        """
        对关系embeddings进行层次聚类

        Args:
            relation_embeddings: [num_relations, embedding_dim] BERT生成的关系embeddings
            method: 聚类连接方法，默认'ward'
            metric: 距离度量，默认'euclidean'

        Returns:
            relation_cluster_labels: [num_relations] 每个关系的簇标签
            relation_cluster_centers: [num_relation_clusters, embedding_dim] 簇中心embeddings
        """
        print("=" * 70)
        print("正在对关系进行层次聚类...")
        print(f"  - 关系数量: {relation_embeddings.shape[0]}")
        print(f"  - 目标簇数: {self.num_relation_clusters}")
        print(f"  - 聚类方法: {method}")
        print(f"  - 距离度量: {metric}")

        # 转换为numpy数组（如果是tensor）
        if isinstance(relation_embeddings, torch.Tensor):
            relation_embeddings_np = relation_embeddings.cpu().numpy()
        else:
            relation_embeddings_np = relation_embeddings

        # 层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=self.num_relation_clusters,
            linkage=method,
            metric=metric
        )

        self.relation_cluster_labels = clustering.fit_predict(relation_embeddings_np)

        # 计算簇中心
        self.relation_cluster_centers = self._compute_cluster_centers(
            relation_embeddings_np,
            self.relation_cluster_labels,
            self.num_relation_clusters
        )

        # 统计每个簇的大小
        cluster_sizes = np.bincount(self.relation_cluster_labels)
        print(f"✓ 关系聚类完成")
        print(f"  - 簇大小统计: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
              f"mean={cluster_sizes.mean():.2f}, std={cluster_sizes.std():.2f}")
        print("=" * 70)

        return self.relation_cluster_labels, self.relation_cluster_centers

    def _compute_cluster_centers(self, embeddings, labels, num_clusters):
        """
        计算簇中心（簇内所有embeddings的平均值）

        Args:
            embeddings: [num_items, embedding_dim] 所有item的embeddings
            labels: [num_items] 每个item的簇标签
            num_clusters: 簇数量

        Returns:
            cluster_centers: [num_clusters, embedding_dim] 簇中心embeddings
        """
        cluster_centers = np.zeros((num_clusters, embeddings.shape[1]))

        for cluster_id in range(num_clusters):
            # 找到属于该簇的所有items
            cluster_mask = (labels == cluster_id)
            cluster_members = embeddings[cluster_mask]

            if len(cluster_members) > 0:
                # 计算平均值作为簇中心
                cluster_centers[cluster_id] = cluster_members.mean(axis=0)
            else:
                # 如果簇为空（理论上不应该发生），使用随机初始化
                print(f"警告: 簇 {cluster_id} 为空，使用随机初始化")
                cluster_centers[cluster_id] = np.random.randn(embeddings.shape[1])

        return cluster_centers

    def save_clustering_results(self, save_dir):
        """
        保存聚类结果到文件

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存实体聚类结果
        entity_labels_path = join(save_dir, 'entity_cluster_labels.npy')
        entity_centers_path = join(save_dir, 'entity_cluster_centers.npy')

        np.save(entity_labels_path, self.entity_cluster_labels)
        np.save(entity_centers_path, self.entity_cluster_centers)

        # 保存关系聚类结果
        relation_labels_path = join(save_dir, 'relation_cluster_labels.npy')
        relation_centers_path = join(save_dir, 'relation_cluster_centers.npy')

        np.save(relation_labels_path, self.relation_cluster_labels)
        np.save(relation_centers_path, self.relation_cluster_centers)

        # 保存聚类配置
        config = {
            'num_entity_clusters': self.num_entity_clusters,
            'num_relation_clusters': self.num_relation_clusters,
            'embedding_dim': self.embedding_dim,
            'num_entities': len(self.entity_cluster_labels),
            'num_relations': len(self.relation_cluster_labels)
        }

        config_path = join(save_dir, 'clustering_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        print(f"✓ 聚类结果已保存到: {save_dir}")
        print(f"  - 实体标签: {entity_labels_path}")
        print(f"  - 实体簇中心: {entity_centers_path}")
        print(f"  - 关系标签: {relation_labels_path}")
        print(f"  - 关系簇中心: {relation_centers_path}")
        print(f"  - 配置文件: {config_path}")

    def load_clustering_results(self, save_dir):
        """
        从文件加载聚类结果

        Args:
            save_dir: 保存目录

        Returns:
            config: 聚类配置字典
        """
        # 加载配置
        config_path = join(save_dir, 'clustering_config.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"聚类配置文件不存在: {config_path}")

        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # 验证配置
        if config['num_entity_clusters'] != self.num_entity_clusters:
            raise ValueError(f"实体簇数量不匹配: 期望{self.num_entity_clusters}, "
                           f"实际{config['num_entity_clusters']}")

        if config['num_relation_clusters'] != self.num_relation_clusters:
            raise ValueError(f"关系簇数量不匹配: 期望{self.num_relation_clusters}, "
                           f"实际{config['num_relation_clusters']}")

        # 加载实体聚类结果
        entity_labels_path = join(save_dir, 'entity_cluster_labels.npy')
        entity_centers_path = join(save_dir, 'entity_cluster_centers.npy')

        self.entity_cluster_labels = np.load(entity_labels_path)
        self.entity_cluster_centers = np.load(entity_centers_path)

        # 加载关系聚类结果
        relation_labels_path = join(save_dir, 'relation_cluster_labels.npy')
        relation_centers_path = join(save_dir, 'relation_cluster_centers.npy')

        self.relation_cluster_labels = np.load(relation_labels_path)
        self.relation_cluster_centers = np.load(relation_centers_path)

        print(f"✓ 聚类结果已加载:")
        print(f"  - 实体数量: {config['num_entities']}, 簇数: {self.num_entity_clusters}")
        print(f"  - 关系数量: {config['num_relations']}, 簇数: {self.num_relation_clusters}")

        return config

    def get_entity_cluster_id(self, entity_id):
        """
        获取指定实体的簇ID

        Args:
            entity_id: 实体ID或ID列表

        Returns:
            cluster_id: 簇ID或ID列表
        """
        if self.entity_cluster_labels is None:
            raise RuntimeError("实体尚未聚类，请先调用cluster_entities()")

        if isinstance(entity_id, (list, np.ndarray, torch.Tensor)):
            return self.entity_cluster_labels[entity_id]
        else:
            return self.entity_cluster_labels[entity_id]

    def get_relation_cluster_id(self, relation_id):
        """
        获取指定关系的簇ID

        Args:
            relation_id: 关系ID或ID列表

        Returns:
            cluster_id: 簇ID或ID列表
        """
        if self.relation_cluster_labels is None:
            raise RuntimeError("关系尚未聚类，请先调用cluster_relations()")

        if isinstance(relation_id, (list, np.ndarray, torch.Tensor)):
            return self.relation_cluster_labels[relation_id]
        else:
            return self.relation_cluster_labels[relation_id]

    def analyze_clustering_quality(self, embeddings, labels, item_type='entity'):
        """
        分析聚类质量（簇内紧密度和簇间分离度）

        Args:
            embeddings: [num_items, embedding_dim]
            labels: [num_items]
            item_type: 'entity' or 'relation'
        """
        print("=" * 70)
        print(f"分析{item_type}聚类质量...")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        num_clusters = len(np.unique(labels))

        # 计算簇内平均距离（越小越好）
        intra_cluster_distances = []
        for cluster_id in range(num_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_members = embeddings[cluster_mask]

            if len(cluster_members) > 1:
                # 计算簇内所有点对之间的平均距离
                center = cluster_members.mean(axis=0)
                distances = np.linalg.norm(cluster_members - center, axis=1)
                intra_cluster_distances.append(distances.mean())

        avg_intra_distance = np.mean(intra_cluster_distances)

        # 计算簇间平均距离（越大越好）
        if item_type == 'entity':
            centers = self.entity_cluster_centers
        else:
            centers = self.relation_cluster_centers

        inter_cluster_distances = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                inter_cluster_distances.append(dist)

        avg_inter_distance = np.mean(inter_cluster_distances)

        # Silhouette系数的简化版本
        separation_ratio = avg_inter_distance / (avg_intra_distance + 1e-10)

        print(f"  - 簇内平均距离: {avg_intra_distance:.4f} (越小越好)")
        print(f"  - 簇间平均距离: {avg_inter_distance:.4f} (越大越好)")
        print(f"  - 分离比率: {separation_ratio:.4f} (越大越好)")
        print("=" * 70)

        return {
            'intra_distance': avg_intra_distance,
            'inter_distance': avg_inter_distance,
            'separation_ratio': separation_ratio
        }


def perform_clustering(entity_embeddings, relation_embeddings, cluster_cache_dir,
                      num_entity_clusters, num_relation_clusters, regenerate_clustering):
    """
    执行静态聚类流程（简化版，只负责聚类）

    职责：根据BERT embeddings执行聚类或加载聚类缓存
    注意：BERT embeddings的生成/加载由调用方（models.py）负责

    Args:
        entity_embeddings: [num_entities, embedding_dim] 实体的BERT embeddings
        relation_embeddings: [num_relations, embedding_dim] 关系的BERT embeddings
        cluster_cache_dir: 聚类结果缓存目录
        num_entity_clusters: 实体簇数量
        num_relation_clusters: 关系簇数量
        regenerate_clustering: 是否强制重新聚类

    Returns:
        clustering: KGClustering实例
    """
    print("\n" + "-" * 70)
    print("执行静态聚类")
    print("-" * 70)

    # 检查聚类缓存是否存在
    cluster_cache_exists = os.path.exists(join(cluster_cache_dir, 'clustering_config.pkl'))

    # 获取embedding维度
    embedding_dim = entity_embeddings.shape[1]

    # 初始化聚类器
    clustering = KGClustering(
        num_entity_clusters=num_entity_clusters,
        num_relation_clusters=num_relation_clusters,
        embedding_dim=embedding_dim
    )

    # 判断是否需要执行聚类
    if not cluster_cache_exists:
        # 缓存不存在，必须执行聚类
        print(f"⚠ 未找到聚类结果缓存，开始聚类...")
        _perform_clustering_and_save(clustering, entity_embeddings, relation_embeddings, cluster_cache_dir)
        print(f"✓ 聚类完成并保存到: {cluster_cache_dir}")

    elif regenerate_clustering:
        # 缓存存在，但用户要求重新聚类
        print(f"⚠ 检测到聚类结果缓存，但需要重新聚类")
        print(f"  正在重新执行聚类...")
        _perform_clustering_and_save(clustering, entity_embeddings, relation_embeddings, cluster_cache_dir)
        print(f"✓ 聚类已重新执行并保存到: {cluster_cache_dir}")

    else:
        # 缓存存在，直接加载
        print(f"✓ 检测到聚类结果缓存，直接加载...")
        print(f"  加载路径: {cluster_cache_dir}")
        try:
            clustering.load_clustering_results(cluster_cache_dir)
            print(f"✓ 聚类结果加载完成")
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠ 加载聚类缓存失败: {e}")
            print(f"  正在重新执行聚类...")
            _perform_clustering_and_save(clustering, entity_embeddings, relation_embeddings, cluster_cache_dir)
            print(f"✓ 聚类已重新执行并保存")

    print(f"  实体数量: {len(clustering.entity_cluster_labels)}, 实体簇数: {clustering.num_entity_clusters}")
    print(f"  关系数量: {len(clustering.relation_cluster_labels)}, 关系簇数: {clustering.num_relation_clusters}")
    print("-" * 70)

    return clustering


def _generate_bert_embeddings(bert_encoder, data, args):
    """
    生成上下文增强的BERT embeddings

    Args:
        bert_encoder: BERTEncoder实例
        data: Data对象
        args: 参数对象

    Returns:
        entity_embeddings: [num_entities, bert_dim]
        relation_embeddings: [num_relations, bert_dim]
    """
    max_neighbors = getattr(args, 'max_neighbors', 5)
    min_weight = getattr(args, 'min_weight', 0.6)

    print(f"  上下文增强参数: max_neighbors={max_neighbors}, min_weight={min_weight}")

    entity_embeddings, relation_embeddings = bert_encoder.get_context_enhanced_embeddings(
        data=data,
        batch_size=32,
        max_neighbors=max_neighbors,
        min_weight=min_weight
    )

    return entity_embeddings, relation_embeddings


def _save_bert_embeddings(entity_embeddings, relation_embeddings, save_dir):
    """
    保存BERT embeddings到指定目录

    Args:
        entity_embeddings: [num_entities, bert_dim]
        relation_embeddings: [num_relations, bert_dim]
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    entity_path = join(save_dir, 'bert_entity_embeddings.pt')
    relation_path = join(save_dir, 'bert_relation_embeddings.pt')

    torch.save(entity_embeddings, entity_path)
    torch.save(relation_embeddings, relation_path)

    print(f"  ✓ 实体embedding保存: {entity_path}")
    print(f"  ✓ 关系embedding保存: {relation_path}")


def _load_bert_embeddings(load_dir):
    """
    从指定目录加载BERT embeddings

    Args:
        load_dir: 加载目录

    Returns:
        entity_embeddings: [num_entities, bert_dim]
        relation_embeddings: [num_relations, bert_dim]
    """
    entity_path = join(load_dir, 'bert_entity_embeddings.pt')
    relation_path = join(load_dir, 'bert_relation_embeddings.pt')

    if not os.path.exists(entity_path) or not os.path.exists(relation_path):
        raise FileNotFoundError(f"BERT embedding文件不存在: {load_dir}")

    entity_embeddings = torch.load(entity_path)
    relation_embeddings = torch.load(relation_path)

    return entity_embeddings, relation_embeddings


def _perform_clustering_and_save(clustering, entity_embeddings, relation_embeddings, save_dir):
    """
    执行聚类并保存结果

    Args:
        clustering: KGClustering实例
        entity_embeddings: [num_entities, embedding_dim]
        relation_embeddings: [num_relations, embedding_dim]
        save_dir: 保存目录
    """
    # 执行聚类
    print("  正在对实体进行聚类...")
    clustering.cluster_entities(entity_embeddings)

    print("  正在对关系进行聚类...")
    clustering.cluster_relations(relation_embeddings)

    # 分析聚类质量
    print("\n  分析聚类质量:")
    entity_quality = clustering.analyze_clustering_quality(
        entity_embeddings,
        clustering.entity_cluster_labels,
        'entity'
    )
    relation_quality = clustering.analyze_clustering_quality(
        relation_embeddings,
        clustering.relation_cluster_labels,
        'relation'
    )

    # 保存聚类结果
    clustering.save_clustering_results(save_dir)
