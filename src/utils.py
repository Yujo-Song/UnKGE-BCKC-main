# definition of hyperparameters
# set by arguments of training

from os.path import join
import torch
import torch.nn as nn
import numpy as np
import random
import scipy
from tqdm import tqdm


def conf_predict(data_triples, model):
    """The goal of evaluate task is to predict the confidence of triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.

    Returns:
        MAE: Mean absolute error.
        MSE: Mean Square error.
    """
    N = data_triples.shape[0]

    confidence = data_triples[:, 3]
    h = data_triples[:, 0]
    r = data_triples[:, 1]
    t = data_triples[:, 2]

    h = torch.tensor(h, dtype=torch.int64).cuda()
    r = torch.tensor(r, dtype=torch.int64).cuda()
    t = torch.tensor(t, dtype=torch.int64).cuda()
    confidence = torch.tensor(confidence, dtype=torch.float32).cuda()

    head_fused, rel_fused, tail_fused = model.cal_dual_scores(h, r, t)
    pred_score = model.cal_confidence(head_fused, rel_fused, tail_fused)
    confidence = torch.unsqueeze(confidence, dim=-1)

    mse = torch.sum(torch.square(pred_score - confidence)).item()
    mae = torch.sum(torch.absolute(pred_score - confidence)).item()

    # pred_score = pred_score.squeeze()  # 维度压缩
    # MAE_loss = nn.L1Loss(reduction="sum")
    # # MAE = MAE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    # MAE = MAE_loss(pred_score, confidence)
    #
    # MSE_loss = nn.MSELoss(reduction="sum")
    # # MSE = MSE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    # MSE = MSE_loss(pred_score, confidence)

    return mae / N, mse / N

def get_mse_neg(data_triples, data, model, neg_per_positive):
    test_triples = data_triples
    N = test_triples.shape[0]

    # negative samples
    # (score - 0)^2
    all_neg_hn_batch = data.corrupt_batch(test_triples, neg_per_positive, "h")
    all_neg_tn_batch = data.corrupt_batch(test_triples, neg_per_positive, "t")
    neg_hn_batch, neg_rel_hn_batch, \
    neg_t_batch, neg_h_batch, \
    neg_rel_tn_batch, neg_tn_batch \
        = all_neg_hn_batch[:, :, 0].astype(int), \
          all_neg_hn_batch[:, :, 1].astype(int), \
          all_neg_hn_batch[:, :, 2].astype(int), \
          all_neg_tn_batch[:, :, 0].astype(int), \
          all_neg_tn_batch[:, :, 1].astype(int), \
          all_neg_tn_batch[:, :, 2].astype(int)

    neg_hn_batch = torch.tensor(neg_hn_batch, dtype=torch.int64).cuda()
    neg_rel_hn_batch = torch.tensor(neg_rel_hn_batch, dtype=torch.int64).cuda()
    neg_t_batch = torch.tensor(neg_t_batch, dtype=torch.int64).cuda()

    neg_h_batch = torch.tensor(neg_h_batch, dtype=torch.int64).cuda()
    neg_rel_tn_batch = torch.tensor(neg_rel_tn_batch, dtype=torch.int64).cuda()
    neg_tn_batch = torch.tensor(neg_tn_batch, dtype=torch.int64).cuda()

    scores_hn = model.cal_score(neg_hn_batch, neg_rel_hn_batch, neg_t_batch)
    scores_tn = model.cal_score(neg_h_batch, neg_rel_tn_batch, neg_tn_batch)

    mse_hn = torch.sum(torch.mean(torch.square(scores_hn - 0), dim=1)) / N
    mse_tn = torch.sum(torch.mean(torch.square(scores_tn - 0), dim=1)) / N

    mse_neg = (mse_hn.item() + mse_tn.item()) / 2

    mae_hn = torch.sum(torch.mean(torch.absolute(scores_hn - 0), dim=1)) / N
    mae_tn = torch.sum(torch.mean(torch.absolute(scores_tn - 0), dim=1)) / N

    mae_neg = (mae_hn.item() + mae_tn.item()) / 2
    return mae_neg, mse_neg

class IndexScore:
    """
    The score of a tail when h and r is given.
    It's used in the ranking task to facilitate comparison and sorting.
    Print w as 3 digit precision float.
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return "(index: %d, w:%.3f)" % (self.index, self.score)
        return "(%d, %.3f)" % (self.index, self.score)

    def __str__(self):
        return "(index: %d, w:%.3f)" % (self.index, self.score)

def get_fixed_hr(hr_map, n=500):
    hr_map500 = {}
    dict_keys = []
    for h in hr_map.keys():
        for r in hr_map[h].keys():
            dict_keys.append([h, r])

    dict_keys = sorted(dict_keys, key=lambda x: len(hr_map[x[0]][x[1]]), reverse=True)
    dict_final_keys = []

    for i in range(2525):
        dict_final_keys.append(dict_keys[i])

    count = 0
    for i in range(n):
        temp_key = random.choice(dict_final_keys)
        h = temp_key[0]
        r = temp_key[1]
        for t in hr_map[h][r]:
            w = hr_map[h][r][t]
            if hr_map500.get(h) == None:
                hr_map500[h] = {}
            if hr_map500[h].get(r) == None:
                hr_map500[h][r] = {t: w}
            else:
                hr_map500[h][r][t] = w

    for h in hr_map500.keys():
        for r in hr_map500[h].keys():
            count = count + 1

    return hr_map500

def get_t_ranks(h, r, ts, tw, model, data):
    """
    Given some t index, return the ranks for each t
    :return:
    """
    N = data.num_cons()

    h_batch = np.repeat(h, N)
    r_batch = np.repeat(r, N)
    t_batch = np.arange(0, N)


    h = torch.tensor(h_batch, dtype=torch.int64).cuda()
    r = torch.tensor(r_batch, dtype=torch.int64).cuda()
    t = torch.tensor(t_batch, dtype=torch.int64).cuda()

    head_fused, rel_fused, tail_fused = model.cal_dual_scores(h, r, t)
    scores = model.cal_ranking_score(head_fused, rel_fused, tail_fused).cpu().numpy()

    ranks = scipy.stats.rankdata(scores, method='ordinal')[ts]
    ranks = N - ranks + 1 #降序排名

    return ranks

def ndcg_(h, r, tw_truth, model, data):
    """
    Compute nDCG(normalized discounted cummulative gain)
    sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
    :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
    :return:
    """
    # prediction
    ts = [tw.index for tw in tw_truth]
    tw = [tw.score for tw in tw_truth]
    ranks = get_t_ranks(h, r, ts, tw, model, data)

    # linear gain
    gains = np.array([tw.score for tw in tw_truth])
    discounts = np.log2(ranks + 1)
    discounted_gains = gains / discounts
    dcg = np.sum(discounted_gains)  # discounted cumulative gain
    # normalize
    max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
    ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

    # exponential gain
    exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
    exp_discounted_gains = exp_gains / discounts
    exp_dcg = np.sum(exp_discounted_gains)
    # normalize
    exp_max_possible_dcg = np.sum(
        exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
    exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

    return ndcg, exp_ndcg

def mean_ndcg_(hr_map, model, data):
    """
    :param hr_map: {h:{r:{t:w}}}
    :return:
    """
    ndcg_sum = 0.0  # nDCG with linear gain
    exp_ndcg_sum = 0.0  # nDCG with exponential gain
    count = 0.0

    # debug ndcg
    res = []  # [(h,r,tw_truth, ndcg)]

    for h in hr_map:
        for r in hr_map[h]:
            tw_dict = hr_map[h][r]  # {t:w}
            tw_truth = [IndexScore(t, w) for t, w in tw_dict.items()]
            tw_truth.sort(reverse=True)  # descending on w
            ndcg, exp_ndcg = ndcg_(h, r, tw_truth, model, data)  # nDCG with linear gain and exponential gain
            ndcg_sum += ndcg
            exp_ndcg_sum += exp_ndcg
            count += 1

    return ndcg_sum / count, exp_ndcg_sum / count

def link_prediction(data, model):
    # 尾实体预测指标
    tail_mr = 0.0
    tail_mrr = 0.0
    tail_wmr= 0.0
    tail_wmrr = 0.0
    tail_hit_1 = 0.0
    tail_hit_3 = 0.0
    tail_hit_5 = 0.0
    tail_hit_10 = 0.0
    tail_hit_20 = 0.0
    tail_hit_40 = 0.0
    tail_w_sum = 0.0
    tail_count = 0.0

    num_test = data.num_cons()

    data_triples = data.test_triples
    mask = data_triples[:, -1] >= 0.0
    data_triples_filtered = data_triples[mask]

    print("combined link prediction filter number:", data_triples_filtered.shape[0])

    #生成候选实体
    candidate_tails = np.arange(0, num_test)
    candidate_tails_tensor = torch.LongTensor(candidate_tails).cuda()

    print("开始链接预测...")
    for head, relation, tail, w in tqdm(data_triples_filtered):
        # 计算真实三元组的分数
        h = torch.tensor([head], dtype=torch.int64).cuda()
        r = torch.tensor([relation], dtype=torch.int64).cuda()
        t = torch.tensor([tail], dtype=torch.int64).cuda()

        h, r, t = model.cal_dual_scores(h, r, t)
        true_score = model.cal_ranking_score(h, r, t).cpu().numpy()

        # 广播head and relation给所有候选尾实体
        head_broadcast = torch.full((num_test,), int(head)).cuda()
        rel_broadcast = torch.full((num_test,), int(relation)).cuda()

        # 获取所有候选尾实体的分数

        head_fused, rel_fused, tail_fused = model.cal_dual_scores(head_broadcast, rel_broadcast, candidate_tails_tensor)
        tail_candidate_scores = model.cal_ranking_score(head_fused, rel_fused, tail_fused).cpu().numpy()

        tw_dict = data.hr_map[head][relation]
        filter_ts = [t_ for t_, w_ in tw_dict.items() if t_ != tail]
        filter_tail_candidate_scores = tail_candidate_scores.copy()
        filter_tail_candidate_scores[filter_ts] = -np.inf
        tail_better_count = (filter_tail_candidate_scores > true_score).astype(int)
        tail_rank = sum(tail_better_count).squeeze() + 1

        # # 计算尾实体排名（基于原始分数）
        # tail_better_score_count = (tail_candidate_scores > true_score).astype(int)
        # tail_rank = sum(tail_better_score_count).squeeze() + 1
        # tail_rank_w = tail_rank * w

        # 累计尾实体预测指标
        tail_mr += tail_rank
        tail_mrr += 1/tail_rank
        tail_wmr += tail_rank * w
        tail_wmrr += 1/tail_rank * w
        tail_w_sum += w
        tail_count += 1

        if tail_rank <= 1:
            tail_hit_1 += 1
        if tail_rank <= 3:
            tail_hit_3 += 1
        if tail_rank <= 5:
            tail_hit_5 += 1
        if tail_rank <= 10:
            tail_hit_10 += 1
        if tail_rank <= 20:
            tail_hit_20 += w
        if tail_rank <= 40:
            tail_hit_40 += w
    
    avg_mr = tail_mr / tail_count
    avg_mrr = tail_mrr / tail_count
    avg_wmr = tail_wmr / tail_w_sum
    avg_wmrr = tail_wmrr / tail_w_sum
    avg_hit_1 = tail_hit_1 / tail_count
    avg_hit_3 = tail_hit_3 / tail_count
    avg_hit_5 = tail_hit_5 / tail_count
    avg_hit_10 = tail_hit_10 / tail_count
    avg_hit_20 = tail_hit_20 / tail_w_sum
    avg_hit_40 = tail_hit_40 / tail_w_sum

    return avg_mr, avg_mrr, avg_wmr, avg_wmrr, avg_hit_1, avg_hit_3, avg_hit_5, avg_hit_10, avg_hit_20, avg_hit_40


def classify_triples(data, model, confT, plausTs):
    """
    Classify high-confidence relation facts
    :param confT: the threshold of ground truth confidence score
    :param plausTs: the list of proposed thresholds of computed plausibility score
    :return:
    """
    test_triples = data.test_triples

    h_batch = test_triples[:, 0].astype(int)
    r_batch = test_triples[:, 1].astype(int)
    t_batch = test_triples[:, 2].astype(int)
    w_batch = test_triples[:, 3]

    # ground truth
    high_gt = set(np.squeeze(np.argwhere(w_batch > confT)))  # positive
    low_gt = set(np.squeeze(np.argwhere(w_batch <= confT)))  # negative

    P = []
    R = []
    Acc = []

    # prediction
    h = torch.tensor(h_batch, dtype=torch.int64).cuda()
    r = torch.tensor(r_batch, dtype=torch.int64).cuda()
    t = torch.tensor(t_batch, dtype=torch.int64).cuda()

    head_fused, rel_fused, tail_fused = model.cal_dual_scores(h, r, t)
    pred_scores = model.cal_confidence(head_fused, rel_fused, tail_fused).cpu().numpy()

    for pthres in plausTs:

        high_pred = set(np.squeeze(np.argwhere(pred_scores > pthres)).flatten())
        low_pred = set(np.squeeze(np.argwhere(pred_scores <= pthres)).flatten())

        # precision-recall
        TP = high_gt & high_pred  # union intersection
        if len(high_pred) == 0:
            precision = 1
        else:
            precision = len(TP) / len(high_pred)

        recall = len(TP) / len(high_gt)
        P.append(precision)
        R.append(recall)

        # accuracy
        TPTN = (len(TP) + len(low_gt & low_pred))
        accuracy = TPTN / test_triples.shape[0]
        Acc.append(accuracy)

    P = np.array(P)
    R = np.array(R)
    F1 = 2 * np.multiply(P, R) / (P + R)
    Acc = np.array(Acc)

    return P, R, F1, Acc


def predict_top_k_tails(model, k, h, r, data, hr_map):
    n = data.num_cons()

    h_batch = np.repeat(h, n)
    r_batch = np.repeat(r, n)
    t_batch = np.arange(0, n)
    t_indices = t_batch

    # print(t_indices)

    # Get predicted scores for candidate tail entities
    h_tensor = torch.tensor(h_batch, dtype=torch.int64).cuda()
    r_tensor = torch.tensor(r_batch, dtype=torch.int64).cuda()
    t_tensor = torch.tensor(t_batch, dtype=torch.int64).cuda()

    head_fused, rel_fused, tail_fused = model.cal_dual_scores(h_tensor, r_tensor, t_tensor)
    predicted_scores = model.cal_confidence(head_fused, rel_fused, tail_fused).squeeze(dim=1).cpu().numpy()
    print(predicted_scores)

    # 按分数降序排序，获取排序后的索引
    sorted_indices = np.argsort(predicted_scores)[::-1]  # [::-1]实现降序

    # 取前k个结果
    top_k_indices = sorted_indices[:k]
    top_k_t = t_indices[top_k_indices]
    top_k_scores = predicted_scores[top_k_indices]

    print(top_k_indices)

    # 准备最终结果
    results = []
    for t, pred_score in zip(top_k_t, top_k_scores):
        # 检查三元组是否存在于数据集中
        if h in hr_map and r in hr_map[h] and t in hr_map[h][r]:
            original_score = hr_map[h][r][t]  # 正确的取值路径
        else:
            original_score = "N/A"  # 不存在则标记为N/A

        pred_score_scalar = pred_score.item() if isinstance(pred_score, np.ndarray) else pred_score

        print(f"h={h}, r={r}, t={t}, 原始分数={original_score}, 预测分数={pred_score_scalar:.4f}")