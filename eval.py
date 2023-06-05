
import numpy as np
from utils.general import file_uri_reader_processor, list_of_dict_to_dict
from utils.metrics import is_hit, rank, group_pred_data_in_time, get_precision_recall


def eval_basemodel_precision_recall(pred_fn, source_fn, rel_topk, obj_topk, num_last_eval_points=4000):
    pred_data = file_uri_reader_processor(pred_fn)[-num_last_eval_points:]
    source_data = file_uri_reader_processor(source_fn)['data']
    group_pred_by_time = group_pred_data_in_time(pred_data, source_data)
    out = get_precision_recall(group_pred_by_time, rel_topk=rel_topk, obj_topk=obj_topk)
    print(out)


def eval_basemodel(pred_fn, num_last_eval_points=4000, rel_hit_top_k=3, obj_hit_top_k=10):
    pred_data = file_uri_reader_processor(pred_fn)[-num_last_eval_points:]
    pred_dict = list_of_dict_to_dict(pred_data)

    rel_pr_topk = 5
    object_pr_topk = 20

    rel_ranks = rank(pred_dict['relation'], pred_dict['pred_relation'])
    obj_ranks = rank(pred_dict['object'], pred_dict['pred_object'])

    rel_mask = rel_ranks <= rel_pr_topk
    obj_mask = obj_ranks <= object_pr_topk

    rel_mean_rank = np.mean(rel_ranks[rel_ranks <= rel_pr_topk])
    obj_mean_rank = np.mean(obj_ranks[obj_ranks <= object_pr_topk])
    relation_hit_ratio = np.mean(
        is_hit(pred_dict['relation'][rel_mask], pred_dict['pred_relation'][rel_mask], top_n=rel_hit_top_k))
    object_hit_ratio = np.mean(
        is_hit(pred_dict['object'][obj_mask], pred_dict['pred_object'][obj_mask], top_n=obj_hit_top_k))
    print(
        f'Relation top{rel_hit_top_k} ratio: {relation_hit_ratio}\n'
        f'Relation mean rank: {rel_mean_rank}\n'
        f'Object top{obj_hit_top_k} ratio: {object_hit_ratio}\n'
        f'Object mean rank: {obj_mean_rank}\n'
    )


def eval_topk_rerank(pred_fn, num_last_eval_points=4000, rel_hit_top_k=3, obj_hit_top_k=10):
    pred_data = file_uri_reader_processor(pred_fn)[-num_last_eval_points:]
    pred_dict = list_of_dict_to_dict(pred_data)

    rel_ranks = rank(pred_dict['relation'], pred_dict['pred_relation'])
    obj_ranks = rank(pred_dict['object'], pred_dict['pred_object'])

    rel_mean_rank = np.mean(rel_ranks[rel_ranks <= rel_hit_top_k])
    obj_mean_rank = np.mean(obj_ranks[obj_ranks <= obj_hit_top_k])
    print(
        f'Relation mean rank: {rel_mean_rank}\n'
        f'Object mean rank: {obj_mean_rank}\n'
    )


def eval_combination_rerank(pred_fn, topk=100, num_last_eval_points=4000):
    pred_data = file_uri_reader_processor(pred_fn)[-num_last_eval_points:]

    rank_list = []
    hit_count = 0
    for point in pred_data:
        # to numerical
        rel_obj_label = '{0}_{1}'.format(np.sum(point['relation']), np.sum(point['object']))
        pred_rel_obj = {
            '{0}_{1}'.format(np.sum(item['pred_relation']), np.sum(item['pred_object'])): np.sum(item['pred_score'])
            for item in point['pred_rel_obj']}
        sorted_pred_rel_obj_list = sorted(pred_rel_obj.items(), key=lambda item: item[1], reverse=True)[:topk]

        for i, (pair, _) in enumerate(sorted_pred_rel_obj_list):
            if pair == rel_obj_label:
                hit_count += 1
                rank_list.append(i + 1)
                break

    print(f'Hit count: {hit_count}, total: {len(pred_data)}')
    print('Hit ratio:', hit_count / len(pred_data))
    print('Mean rank:', np.mean(rank_list))


if __name__ == '__main__':
    source_fn = 'data/gdelt/gdelt.pkl'
    ke_anhp_gdelt_fn = 'logs/ke_anhp_gdelt_test.pkl'

    eval_basemodel_precision_recall(pred_fn=ke_anhp_gdelt_fn, source_fn=source_fn, rel_topk=5, obj_topk=20)
    # eval_topk_rerank(ke_anhp_gdelt_fn, rel_hit_top_k=10, obj_hit_top_k=2)

