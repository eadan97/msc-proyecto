import logging
import os
import torch


def get_model_size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def calculate_metrics(train, test, model):
    all_predictions = predict_ranking(model, train, usercol='userID', itemcol='itemID', remove_seen=True)
    eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    res = {
        "MAP": eval_map,
        "NDCG": eval_ndcg,
        "Precision@K": eval_precision,
        "Recall@K": eval_recall,
    }
    logging.info(res)
    return res
