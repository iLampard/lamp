
import numpy as np
from utils.general import file_uri_reader_processor, get_value_by_key
from utils.metrics import mark, mapk, time_rmse_np, rank


def eval_basemodel_precision_recall(pred_fn):
    pred_data = file_uri_reader_processor(pred_fn)

    label_dtime = []
    pred_dtime = []
    label_type = []
    pred_type = []


    for seq in pred_data: # by default, we do the inference with batch_size = 1
        label_type_ = get_value_by_key('label_type', seq)
        pred_type_score_ = get_value_by_key('pred_type_score', seq)
        label_dtime_ = get_value_by_key('label_dtime', seq)
        pred_dtime_ = get_value_by_key('pred_dtime', seq)

        label_dtime.extend(label_dtime_)
        pred_dtime.extend(pred_dtime_)

        # each label is a list
        label_type.extend([[x] for x in label_type_])

        # make score in descending order
        pred_type_ = [np.argsort(x)[-5:][::-1].tolist() for x in pred_type_score_]
        pred_type.extend(pred_type_)


    # precision - recall
    print(f'MAP is {mapk(label_type, pred_type, k=5)}')
    print(f'MAR is {mark(label_type, pred_type, k=5)}')

    # compute time rmse
    time_rmse = time_rmse_np(pred_dtime, label_dtime)
    print(f'Time RMSE {time_rmse}')
    return


def eval_base_model_mean_rank(pred_fn, target_events):
    pred_data = file_uri_reader_processor(pred_fn)

    pred_target_data = []
    pred_type_score = []
    label_type = []
    for event in target_events:
        seq_idx, original_idx = eval(event[0])
        pred_event = search_pred_data(pred_data, seq_idx, original_idx)
        pred_target_data.append(pred_event)
        pred_type_score.append(pred_event['pred_type_score'])
        label_type.append(pred_event['label_type'])


    type_pr_topk = 5
    type_ranks = rank(label_type, pred_type_score)
    type_mask = type_ranks <= type_pr_topk

    type_mean_ranks = np.mean(type_ranks[type_mask])
    print(type_mean_ranks)
    return

def search_pred_data(pred_data, seq_idx, original_idx):

    for pred_seq in pred_data:
        if pred_seq[0]['seq_idx'] == int(seq_idx):
            for pred_element in pred_seq:
                if pred_element['original_idx'] == int(original_idx):
                    return pred_element

    print('Error, index not found')

if __name__ == '__main__':
    anhp_amazon_fn = 'logs/tpp_amazon_test.pkl'

    import dictdatabase as DDB
    DDB.config.storage_directory = 'scripts/amazon/ddb_storage'
    ebm_data = list(DDB.at(f'anhp_amazon_bert_ebm_dataset', 'type').read().values())

    eval_base_model_mean_rank(anhp_amazon_fn, ebm_data)