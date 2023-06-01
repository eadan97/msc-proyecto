import logging
import os
import torch

import numpy as np

from recommenders.models.cornac.cornac_utils import predict_ranking, predict
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.timer import Timer

TOP_K = 10

def get_model_size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def calculate_metrics(train, test, model, usercol='userID', itemcol='itemID'):
    all_predictions = predict_ranking(model, train, usercol=usercol, itemcol=itemcol, remove_seen=True)
    # eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    # eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    res = {
        # "MAP": eval_map,
        "NDCG": eval_ndcg,
        # "Precision@K": eval_precision,
        "Recall@K": eval_recall,
    }
    logging.info(res)
    return res
