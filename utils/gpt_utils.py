import pickle
from datetime import datetime
from datetime import timedelta

import numpy as np


def get_entity_map(pkl_dir):
    with open(pkl_dir, 'rb') as f:
        entity_map = pickle.load(f)
    return entity_map


try:
    AMAZON_TYPE_MAP = get_entity_map('../../data/gdelt/amazon_type_map.pkl')
except:
    print('please use amazon type map pkl!')
    pass

ENTITY_MAP = get_entity_map('../../data/gdelt/entity_map.pkl')

EVENT_TYPE_MAP = {0: 'MAKE STATEMENT',
                  1: 'APPEAL',
                  2: 'EXPRESS INTENT TO COOPERATE',
                  3: 'CONSULT',
                  4: 'ENGAGE IN DIPLOMATIC COOPERATION',
                  5: 'ENGAGE IN MATERIAL COOPERATION',
                  6: 'PROVIDE AID',
                  7: 'YIELD',
                  8: 'INVESTIGATE',
                  9: 'DEMAND',
                  10: 'DISAPPROVE',
                  11: 'REJECT',
                  12: 'THREATEN',
                  13: 'PROTEST',
                  14: 'EXHIBIT MILITARY POSTURE',
                  15: 'REDUCE RELATIONS',
                  16: 'COERCE',
                  17: 'ASSAULT',
                  18: 'FIGHT',
                  19: 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE'}


def generate_gdelt_prompt_from_pred_relation(model_output, event_seq, top_k: int = 5):
    """Generate the prompt that describes the effect event.
    """
    # find the top-k relations
    pred_rel_list = model_output['pred_relation']

    topk_rels = np.argsort(pred_rel_list)[-top_k:]
    original_idx = model_output['original_idx']
    original_event = event_seq[original_idx]
    label_rel = original_event[2]

    all_rels = set(topk_rels.tolist() + [label_rel])
    parts = original_event[-1].split(';')
    prompt_dict = {
        rel: 'effect event\n' +
             'event type:' + EVENT_TYPE_MAP[rel] + '\n' +
             parts[-3].strip() + '\n' +
             parts[-2].strip() + '\n' +
             parts[-1].strip() + '\n'
        for rel in all_rels
    }
    return prompt_dict


def generate_gdelt_prompt_v2(model_output, event_seq, top_k: int = 5, pred_relation: bool = True):
    """Generate the prompt that describes the effect event.
    """
    # find the top-k relations
    if pred_relation:
        pred_list = model_output['pred_relation']
    else:
        pred_list = model_output['pred_object']

    if len(pred_list.shape) > 1:
        pred_list = pred_list[0, :]

    top_k = np.argsort(pred_list)[-top_k:]
    original_idx = model_output['original_idx']
    original_event = event_seq[original_idx]

    if pred_relation:
        label = original_event[2]
    else:
        label = original_event[1]  # object

    all_preds = set(top_k.tolist() + [label])
    parts = original_event[-1].split(';')

    if pred_relation:
        prompt_dict = {
            rel: 'effect event\n' +
                 'event type:' + EVENT_TYPE_MAP[rel] + '\n' +
                 parts[-3].strip() + '\n' +
                 parts[-2].strip() + '\n' +
                 parts[-1].strip() + '\n'
            for rel in all_preds
        }
    else:
        prompt_dict = {
            obj: 'effect event\n' +
                 parts[-4].strip() + '\n' +
                 parts[-3].strip() + '\n' +
                 parts[-2].strip() + '\n' +
                 'object name:' + ENTITY_MAP[obj] + '\n'
            for obj in all_preds
        }
    return prompt_dict


def generate_gdelt_comb_prompt(model_output, event_seq, top_k: int = 5, filter_hit=True):
    # find the top-k relations
    pred_rel_obj_dict = {
        '{0}_{1}'.format(np.sum(item['pred_relation']), np.sum(item['pred_object'])): np.sum(item['pred_score'])
        for item in model_output['pred_rel_obj']}
    pred_rel_obj_topk_list = sorted(pred_rel_obj_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
    pred_rel_obj_topk = [pair.split('_') for pair, _ in pred_rel_obj_topk_list]

    original_idx = model_output['original_idx']
    original_event = event_seq[original_idx]

    is_hit = '{0}_{1}'.format(original_event[2], original_event[1]) in pred_rel_obj_dict

    if not is_hit:
        if filter_hit:
            return {}
        else:
            pred_rel_obj_topk.append([str(original_event[2]), str(original_event[1])])

    parts = original_event[-1].split(';')
    prompt_dict = {
        rel + '_' + obj: 'effect event\n' +
                         'event type:' + EVENT_TYPE_MAP[int(rel)] + '\n' +
                         parts[-3].strip() + '\n' +
                         parts[-2].strip() + '\n' +
                         'object name:' + ENTITY_MAP[int(obj)] + '\n'
        for rel, obj in pred_rel_obj_topk
    }
    return prompt_dict


def get_dtime_top_n(seq_dtimes, num_bins: int = 50, top_k: int = 5):
    hist, bin_edges = np.histogram(seq_dtimes, bins=num_bins)
    top_k_index = np.argsort(hist)[-top_k:]
    return bin_edges[top_k_index]


def generate_gdelt_prompt_amazon(model_output, event_seq, top_k: int = 5, pred_type: bool = True):
    """Generate the prompt that describes the effect event.
    """
    seq_idx = model_output['seq_idx']
    original_idx = model_output['original_idx']
    if pred_type:
        # find the top-k relations
        pred_list = model_output['pred_type_score']
        topk_pred = np.argsort(pred_list)[-top_k:]
        label_type = event_seq[seq_idx][original_idx]['event_type']
        all_types = set(topk_pred.tolist() + [label_type])
    else:
        # seq_dtimes = [x['event_dtime'] for x in event_seq[seq_idx][1:original_idx]]
        # have to add mbr here
        # all_dtimes = get_dtime_top_n(seq_dtimes, top_k=top_k).tolist()
        # all_dtimes.append(model_output['pred_dtime'])
        all_dtimes = model_output['pred_dtime']
        prev_date = event_seq[seq_idx][original_idx - 1]['event_date']
        prev_date = datetime.strptime(prev_date, "%Y-%m-%d")
        all_date = [prev_date + timedelta(days=int(dtime)) for dtime in all_dtimes]
        # revert to string
        all_date = [date.strftime("%Y-%m-%d") for date in all_date]
        label_date = event_seq[seq_idx][original_idx]['event_date']
        all_date = set(all_date + [label_date])

    parts = event_seq[seq_idx][original_idx]['event_text'].split(';')

    if pred_type:
        prompt_dict = {
            type: 'effect event\n' +
                  'product category:' + AMAZON_TYPE_MAP[type] + '\n' +
                  parts[-3].strip() + '\n'  # event time
            for type in all_types
        }
    else:
        prompt_dict = {
            date: 'effect event\n' +
                  parts[0].strip() + '\n' +
                  'event time:' + date + '\n'  # event time
            for date in all_date
        }
    return prompt_dict
