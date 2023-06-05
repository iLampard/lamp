import numpy as np
from Levenshtein import distance

from utils.general import ProcessPool


def parse_causal_event_from_text_amazon(llm_text_output):
    import re

    product_category = re.compile(r"product category:[\s]*(.+)[\s]*").findall(llm_text_output)
    product_title = re.compile(r"product title:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_times = re.compile(r"event time:[\s]*(.+)[\s]*").findall(llm_text_output)
    summary_text = re.compile(r"summary text:[\s]*(.+)[\s]*").findall(llm_text_output)
    review_text = re.compile(r"review text:[\s]*(.+)[\s]*").findall(llm_text_output)

    sample_list = []
    for category, title, time, summary, review in zip(product_category, product_title, event_times, summary_text,
                                                      review_text):
        one = ';'.join([
            'product category:' + category,
            'product title:' + title,
            'event time:' + time,
            'summary text:' + summary,
            'review text:' + review,
        ])
        sample_list.append(one)

    return sample_list


def parse_events(model_output):
    res = dict()
    for event_type, prompt_output_str in model_output.items():
        res[event_type] = parse_causal_event_from_text_amazon(prompt_output_str)
    return res


def make_event_dict(event_list):
    return {'product category': event_list[0],
            'event time': event_list[2],
            'seq_index': event_list[3],
            'original_index': event_list[4]}


def retrieve_event_from_source_data_amazon(target_event, source_data, seq_index, original_index, top_n=1):
    # search over the previous events
    # fix to search over the last 10k events
    # source_data_ = source_data[int(original_index) - context_window:int(original_index)]
    target_seq = source_data[seq_index]
    source_data_ = target_seq[:original_index]
    # scores = [levenshteinDistance(target_event, event[4]) for event in source_data_]
    scores = [distance(target_event, event[4]) for event in source_data_]
    sort_index = np.argsort(np.array(scores))
    retro_events = []
    # to do list selection, we need to convert source data to np.array, which may be slow
    # so i do the list append here.
    select_index = sort_index[:top_n]
    for i in select_index:
        retro_events.append(make_event_dict(source_data_[i]))
    return retro_events


def get_event(source_data, seq_idx, original_idx):
    seq = source_data[seq_idx]
    event = (seq['event_type'][original_idx], seq['event_time'][original_idx], seq_idx, original_idx)
    return event


def is_event_existed(target_event, event_list):
    for event in event_list:
        s = 0
        for k, v in target_event.items():
            s += abs(target_event[k] - event[k])
        if s < 0.001:
            return True
    return False


def make_complete_event_sequence(real_event, retro_event_dict, pred_name='rel'):
    """ Add the last event to the samples """

    event_type, event_time, seq_idx, original_idx = real_event

    real_event_seq_sample = None
    noise_event_seq_sample = []
    noise_event_target = []
    real_event_target = label_rel
    for pred_target, causal_evts in retro_event_dict.items():
        if find_label(pred_target, real_event, pred_name):
            real_event_seq_sample = causal_evts
            # append the last event
            real_event_seq_sample.append(make_event_dict(real_event))
        else:
            noise_seq_ = causal_evts
            noise_seq_.append(make_event_dict(event_time, seq_idx, original_idx))
            noise_event_target.append(int(pred_target))
            noise_event_seq_sample.append(noise_seq_)

    return real_event_seq_sample, noise_event_seq_sample, real_event_target, noise_event_target


def make_samples_for_energy_function_amazon(db, source_data, retro_top_n=2):
    import dictdatabase as DDB

    ebm_db_name = 'ebm_dataset_amazon'

    def _process(s_idx, o_idx, p_e_d):
        retro_event_dict = {}
        for pred_type, prompt_event_list in p_e_d.items():
            retro_event_list = [
                retrieve_event_from_source_data_amazon(
                    target_event=prompt_event,
                    source_data=source_data,
                    seq_index=s_idx,
                    original_index=o_idx,
                    top_n=retro_top_n)
                for prompt_event in prompt_event_list
            ]

            # do a flatten
            retro_event_list = [item for sublist in retro_event_list for item in sublist]

            # sort by time and drop duplicate
            retro_event_list_ = []
            for i, event in enumerate(retro_event_list):
                if i == 0:
                    retro_event_list_.append(event)
                    continue
                else:
                    if not is_event_existed(event, retro_event_list_):
                        retro_event_list_.append(event)

            retro_event_list_ = sorted(retro_event_list_, key=lambda x: x['event_time'])
            # save
            retro_event_dict[pred_type] = retro_event_list_

        real_event_ = get_event(source_data, seq_idx=seq_index, original_idx=original_index)
        real_event_sample, noise_event_sample, real_type, noise_type = make_complete_event_sequence(real_event_,
                                                                                                    retro_event_dict)

        is_label_in_top_5 = list(p_e_d.keys()) == 5
        res_i = (
            o_idx,
            real_event_sample,
            noise_event_sample,
            is_label_in_top_5,
            real_type,
            noise_type
        )
        with DDB.at(ebm_db_name).session() as (sess, ebm_db):
            ebm_db[(str(s_idx), str(o_idx))] = res_i
            sess.write()
        print('--- ok', s_idx, o_idx)
        # return res_i

    if not DDB.at(ebm_db_name).exists():
        DDB.at(ebm_db_name).create()

    existed_ebm_db_dict = DDB.at(ebm_db_name).read()

    dp_list = []
    for k, v in db.db.items():
        if k in existed_ebm_db_dict and len(existed_ebm_db_dict[k][2]) >= 5:
            continue
        if len(v) == 0:
            continue

        seq_index, original_index = k

        # parse the prompt
        res_i_prompt_event_dict = parse_events(v)
        if len(res_i_prompt_event_dict) == 0:
            continue

        dp_list.append(
            (seq_index, original_index, res_i_prompt_event_dict)
        )

    with ProcessPool() as pool:
        pool.run(
            target=_process,
            dynamic_param_list=dp_list
        )
    # file_uri_writer_processor(res_list, 'ebm_dataset_v0326.pkl')
