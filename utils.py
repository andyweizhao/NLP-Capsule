import numpy as np
from multiprocessing import Pool
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def get_result(args):

    (y_pred, y_true)=args

    top_k = 50
    pred_topk_index = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)[:top_k]
    pos_index = set([k for k, v in enumerate(y_true) if v == 1])

    r = [1 if k in pos_index else 0 for k in pred_topk_index[:top_k]]

    p_1 = precision_at_k(r, 1)
    p_3 = precision_at_k(r, 3)
    p_5 = precision_at_k(r, 5)

    ndcg_1 = ndcg_at_k(r, 1)
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)

    return np.array([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5])

def evaluate(Y_tst_pred, Y_tst):
    pool = Pool(12)
    results = pool.map(get_result,zip(list(Y_tst_pred), list(Y_tst)))
    pool.terminate()
    tst_result = list(np.mean(np.array(results),0))
    print ('\rTst Prec@1,3,5: ', tst_result[:3], ' Tst NDCG@1,3,5: ', tst_result[3:])
