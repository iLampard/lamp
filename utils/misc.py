import bisect
import copy
import os
import pickle

import numpy as np
import yaml
from Levenshtein import distance
from numpy.linalg import norm

from utils.general import ProcessPool


def py_assert(condition, exception_type, msg):
    """An assert function that ensures the condition holds, otherwise throws a message.

    Args:
        condition (bool): a formula to ensure validity.
        exception_type (_StandardError): Error type, such as ValueError.
        msg (str): a message to throw out.

    Raises:
        exception_type: throw an error when the condition does not hold.
    """
    if not condition:
        raise exception_type(msg)


def make_config_string(config, max_num_key=4):
    """Generate a name for config files.

    Args:
        config (dict): configuration dict.
        max_num_key (int, optional): max number of keys to concat in the output. Defaults to 4.

    Returns:
        dict: a concatenated string from config dict.
    """
    str_config = ''
    num_key = 0
    for k, v in config.items():
        if num_key < max_num_key:  # for the moment we only record model name
            if k == 'name':
                str_config += str(v) + '_'
                num_key += 1
    return str_config[:-1]


def save_yaml_config(save_dir, config):
    """A function that saves a dict of config to yaml format file.

    Args:
        save_dir (str): the path to save config file.
        config (dict): the target config object.
    """
    prt_dir = os.path.dirname(save_dir)

    from collections import OrderedDict
    # add yaml representer for different type
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    )

    if prt_dir != '' and not os.path.exists(prt_dir):
        os.makedirs(prt_dir)

    with open(save_dir, 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    return


def load_yaml_config(config_dir):
    """ Load yaml config file from disk.

    Args:
        config_dir: str or Path
            The path of the config file.

    Returns:
        Config: dict.
    """
    with open(config_dir) as config_file:
        # load configs
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config


def create_folder(*args):
    """Create path if the folder doesn't exist.

    Returns:
        str: the created folder's path.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_pickle(file_dir):
    """Load from pickle file.

    Args:
        file_dir (BinaryIO): dir of the pickle file.

    Returns:
        any type: the loaded data.
    """
    try:
        data = pickle.load(file_dir, encoding='latin-1')
    except Exception:
        data = pickle.load(file_dir)

    return data


def save_pickle(file_dir, object_to_save):
    """Save the object to a pickle file.

    Args:
        file_dir (str): dir of the pickle file.
        object_to_save (any): the target data to be saved.
    """

    with open(file_dir, "wb") as f_out:
        pickle.dump(object_to_save, f_out)

    return


def has_key(target_dict, target_keys):
    """Check if the keys exist in the target dict.

    Args:
        target_dict (dict): a dict.
        target_keys (str, list): list of keys.

    Returns:
        bool: True if all the key exist in the dict; False otherwise.
    """
    if not isinstance(target_keys, list):
        target_keys = [target_keys]
    for k in target_keys:
        if k not in target_dict:
            return False
    return True


def array_pad_cols(arr, max_num_cols, pad_index):
    """Pad the array by columns.

    Args:
        arr (np.array): target array to be padded.
        max_num_cols (int): target num cols for padded array.
        pad_index (int): pad index to fill out the padded elements

    Returns:
        np.array: the padded array.
    """
    res = np.ones((arr.shape[0], max_num_cols)) * pad_index

    res[:, :arr.shape[1]] = arr

    return res


def concat_element(arrs, pad_index):
    """ Concat element from each batch output  """

    n_lens = len(arrs)
    n_elements = len(arrs[0])

    # found out the max seq len (num cols) in output arrays
    max_len = max([x[0].shape[1] for x in arrs])

    concated_outputs = []
    for j in range(n_elements):
        a_output = []
        for i in range(n_lens):
            arrs_ = array_pad_cols(arrs[i][j], max_num_cols=max_len, pad_index=pad_index)
            a_output.append(arrs_)

        concated_outputs.append(np.concatenate(a_output, axis=0))

    # n_elements * [ [n_lens, dim_of_element] ]
    return concated_outputs


def to_dict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = to_dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__"):
        return [to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, to_dict(value, classkey))
                     for key, value in obj.__dict__.iteritems()
                     if not callable(value) and not key.startswith('_') and key not in ['name']])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def dict_deep_update(target, source, is_add_new_key=True):
    """ Update 'target' dict by 'source' dict deeply, and return a new dict copied from target and source deeply.

    Args:
        target: dict
        source: dict
        is_add_new_key: bool, default True.
            Identify if add a key that in source but not in target into target.

    Returns:
        New target: dict. It contains the both target and source values, but keeps the values from source when the key
        is duplicated.
    """
    # deep copy for avoiding to modify the original dict
    result = copy.deepcopy(target) if target is not None else {}

    if source is None:
        return result

    for key, value in source.items():
        if key not in result:
            if is_add_new_key:
                result[key] = value
            continue
        # both target and source have the same key
        base_type_list = [int, float, str, tuple, bool, list]
        if type(result[key]) in base_type_list or type(source[key]) in base_type_list:
            result[key] = value
        else:
            result[key] = dict_deep_update(result[key], source[key], is_add_new_key=is_add_new_key)
    return result


def parse_causal_event_from_text(llm_text_output):
    import re

    headlines = re.compile(r"event headline:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_types = re.compile(r"event type:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_times = re.compile(r"event time:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_subjects = re.compile(r"subject name:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_objects = re.compile(r"object name:[\s]*(.+)[\s]*").findall(llm_text_output)

    sample_list = []
    for sub, obj, rel, time, hl in zip(event_subjects, event_objects, event_types, event_times, headlines):
        one = ';'.join([
            'event headline:' + hl,
            'event type:' + rel,
            'event time:' + time,
            'subject name:' + sub,
            'object name:' + obj,
        ])
        sample_list.append(one)

    return sample_list


def parse_causal_event_from_text_amazon(llm_text_output):
    import re

    product_category = re.compile(r"product category:[\s]*(.+)[\s]*").findall(llm_text_output)
    product_title = re.compile(r"product title:[\s]*(.+)[\s]*").findall(llm_text_output)
    event_time = re.compile(r"event time:[\s]*(.+)[\s]*").findall(llm_text_output)
    summary_text = re.compile(r"summary text:[\s]*(.+)[\s]*").findall(llm_text_output)
    review_text = re.compile(r"review text:[\s]*(.+)[\s]*").findall(llm_text_output)

    sample_list = []
    for product_category_, product_title_, event_time_, \
        summary_text_, review_text_ in zip(product_category, product_title, event_time, summary_text, review_text):
        one = ';'.join([
            'product category:' + product_category_,
            'product title:' + product_title_,
            'event time:' + event_time_,
            'summary text:' + summary_text_,
            'review text:' + review_text_,
        ])
        sample_list.append(one)

    return sample_list


def parse_events(model_output, data_name='gdelt'):
    res = dict()
    for rel, prompt_output_str in model_output.items():
        if data_name == 'gdelt':
            res[rel] = parse_causal_event_from_text(prompt_output_str)
        else:
            res[rel] = parse_causal_event_from_text_amazon(prompt_output_str)
    return res


def make_event_dict(event_list, data_name='gdelt'):
    if data_name == 'gdelt':
        return {'subject': event_list[0], 'object': event_list[1], 'relation': event_list[2], 'time': event_list[3]}
    else:
        return make_json_serializable(event_list)


def get_end_index_by_find_prev_time(total_times, target_time):
    idx = bisect.bisect_left(total_times, target_time)
    return idx


def get_total_time_list(source_data):
    time_list = [x[3] for x in source_data]
    return time_list


def event_distance(text1, text2, distance_type='edit'):
    if distance_type == 'edit':
        return distance(text1, text2)
    else:
        return np.dot(text1, text2) / (norm(text1) * norm(text2))


def retrieve_event_from_source_data(target_event, time_list, source_data, original_index, top_n=1, data_name='gdelt',
                                    distance_type='edit', encoder_model=None):
    if data_name == 'gdelt':
        # search over the previous events
        # fix to search over the last 10k events
        # source_data_ = source_data[int(original_index) - context_window:int(original_index)]
        target_time = source_data[int(original_index)][3]
        retrieve_idx = get_end_index_by_find_prev_time(time_list, target_time)

        source_data_ = source_data[:retrieve_idx]
        # source_data_ = source_data_[-50000:]
        # scores = [levenshteinDistance(target_event, event[4]) for event in source_data_]
        if distance_type == 'edit':
            scores = [distance(target_event, event[4]) for event in source_data_]
        elif distance_type == 'bert':
            target_text = encoder_model.encode(target_event)
            scores = [event_distance(target_text, event[-1], distance_type=distance_type) for event in source_data_]
        else:
            target_text = encoder_model.encode(target_event, padding="max_length", max_length=60, truncation=True)
            scores = [event_distance(target_text, event[-1], distance_type=distance_type) for event in source_data_]
    else:
        seq_idx, pos_idx = eval(original_index)
        source_data_ = source_data[int(seq_idx)][:int(pos_idx)]
        scores = [distance(target_event, event['event_text']) for event in source_data_]

    sort_index = np.argsort(np.array(scores))
    retro_events = []
    # to do list selection, we need to convert source data to np.array, which may be slow
    # so i do the list append here.
    select_index = sort_index[:top_n]
    for i in select_index:
        retro_events.append(make_event_dict(source_data_[i], data_name=data_name))
    return retro_events


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def is_event_existed(target_event, event_list, target_keys=None):
    target_keys = target_event.keys() if target_keys is None else target_keys
    for event in event_list:
        s = 0
        for k in target_keys:
            s += abs(target_event[k] - event[k])
        if s < 0.001:
            return True
    return False


def make_json_serializable(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, np.float64):
            input_dict[k] = float(v)
        if isinstance(v, np.int64):
            input_dict[k] = int(v)

    return input_dict


def make_complete_event_sequence(real_event, retro_event_dict, pred_type='rel', data_name='gdelt'):
    """ Add the last event to the samples """

    def find_label(pred_target, real_event, pred_type, data_name):
        if data_name == 'gdelt':
            if pred_type == 'relation':
                label = str(real_event[2])
                return pred_target == str(label)
            elif pred_type == 'rel_obj':
                pred_rel, pred_obj = pred_target.split('_')
                return int(pred_rel) == int(real_event[2]) and int(pred_obj) == int(real_event[1])
        else:
            if pred_type == 'type':
                label = real_event['event_type']
                return pred_target == str(label)
            else:
                label = real_event['event_dtime']
                return pred_target == str(label)

    sub, label_obj, label_rel, time = real_event[:4]

    real_event_seq_sample = None
    noise_event_seq_sample = []
    noise_event_target = []

    if pred_type == 'relation':
        real_event_target = label_rel
    elif pred_type == 'object':
        real_event_target = label_obj
    else:
        real_event_target = '{0}_{1}'.format(label_rel, label_obj)
    for pred_target, causal_evts in retro_event_dict.items():
        if find_label(pred_target, real_event, pred_type, data_name=data_name):
            real_event_seq_sample = causal_evts
            # append the last event
            real_event_seq_sample.append(make_event_dict(real_event, data_name=data_name))
        else:
            noise_seq_ = causal_evts

            if pred_type == 'relation':
                noise_seq_.append(make_event_dict((sub, label_obj, int(pred_target), time)))
            elif pred_type == 'rel_obj':
                pred_rel, pred_obj = pred_target.split('_')
                noise_seq_.append(make_event_dict((sub, int(pred_obj), int(pred_rel), time)))
            else:
                noise_seq_.append(make_event_dict((sub, int(pred_target), label_rel, time)))
            noise_event_target.append(pred_target)
            noise_event_seq_sample.append(noise_seq_)

    return real_event_seq_sample, noise_event_seq_sample, real_event_target, noise_event_target


def make_samples_for_energy_function(
        gpt_db_name,
        source_data,
        pred_data,
        topk=5,
        pred_type='relation',
        ebm_db_name=None,
        retro_top_n=2,
        distance_type='edit'
):
    import dictdatabase as DDB

    encoder_model = None
    if distance_type == 'bert':
        from sentence_transformers import SentenceTransformer
        encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
    elif distance_type == 'sparse':
        from transformers import AutoTokenizer
        encoder_model = AutoTokenizer.from_pretrained("gpt2")
        encoder_model.pad_token = encoder_model.eos_token

    def _process(o_idx, p_e_d):
        retro_event_dict = {}
        # retrieve the close events
        total_time_list = get_total_time_list(source_data)

        for pred_rel, prompt_event_list in p_e_d.items():
            retro_event_list = [
                retrieve_event_from_source_data(
                    target_event=prompt_event,
                    time_list=total_time_list,
                    source_data=source_data,
                    original_index=o_idx,
                    top_n=retro_top_n,
                    distance_type=distance_type,
                    encoder_model=encoder_model)
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

            retro_event_list_ = sorted(retro_event_list_, key=lambda x: x['time'])
            # save
            retro_event_dict[pred_rel] = retro_event_list_

        real_event_ = source_data[int(o_idx)]
        real_event_sample, noise_event_sample, real_rel, noise_rel = make_complete_event_sequence(real_event_,
                                                                                                  retro_event_dict,
                                                                                                  pred_type=pred_type)

        res_i = (
            o_idx,
            real_event_sample,
            noise_event_sample,
            real_rel,
            noise_rel
        )

        with DDB.at(ebm_db_name, pred_type).session() as (sess, ebm_db):
            ebm_db[str(o_idx)] = res_i
            sess.write()
        print('--- ok', o_idx)
        # return res_i

    if not DDB.at(ebm_db_name, pred_type).exists():
        DDB.at(ebm_db_name, pred_type).create()

    existed_ebm_db_dict = DDB.at(ebm_db_name, pred_type).read()

    dp_list = []
    gpt_db_dict = DDB.at(gpt_db_name).read()
    for pred_tuple in pred_data:
        str_idx = str(pred_tuple['original_idx'])
        thresh = topk - 1
        if str_idx in existed_ebm_db_dict and len(existed_ebm_db_dict[str_idx][2]) >= thresh:
            continue
        topk_pred = np.argsort(pred_tuple[f'pred_{pred_type}'])[-topk:]

        if str_idx not in gpt_db_dict:
            print(f'Idx {str_idx} is not in gpt_db')
            continue

        # just retrieve and add the label event
        topk_causal_dict = {}
        for pred in set(pred_tuple[pred_type].tolist() + topk_pred.tolist()):
            if str(pred) not in gpt_db_dict[str_idx]:
                print(f'-Miss {str_idx}-{pred} in gpt_db')
                continue
            topk_causal_dict[str(pred)] = gpt_db_dict[str_idx][str(pred)]

        if len(topk_causal_dict) < topk:
            print(f'Causal events of the Idx {str_idx} are not fully fetched in gpt_db')
            continue

        res_i_prompt_event_dict = parse_events(topk_causal_dict)
        if len(res_i_prompt_event_dict) == 0:
            print(f'Idx {str_idx} can not be parse correctly')
            continue

        dp_list.append(
            (str_idx, res_i_prompt_event_dict)
        )

    with ProcessPool() as pool:
        pool.run(
            target=_process,
            dynamic_param_list=dp_list
        )


def make_samples_for_energy_function_amazon(
        gpt_db_name,
        source_data,
        pred_data,
        topk=5,
        pred_type='type',
        ebm_db_name=None,
        retro_top_n=2,
        data_name='amazon',
        distance_type='edit'
):
    import dictdatabase as DDB

    encoder_model = None
    if distance_type == 'bert':
        from sentence_transformers import SentenceTransformer
        encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
    elif distance_type == 'sparse':
        from transformers import AutoTokenizer
        encoder_model = AutoTokenizer.from_pretrained("gpt2")
        encoder_model.pad_token = encoder_model.eos_token

    def _process(o_idx, p_e_d):
        retro_event_dict = {}

        for pred_rel, prompt_event_list in p_e_d.items():
            retro_event_list = [
                retrieve_event_from_source_data(
                    target_event=prompt_event,
                    time_list=None,
                    source_data=source_data,
                    original_index=o_idx,
                    top_n=retro_top_n,
                    data_name=data_name,
                    distance_type=distance_type,
                    encoder_model=encoder_model)
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
                    if not is_event_existed(event, retro_event_list_, ['event_time', 'event_type']):
                        retro_event_list_.append(event)

            retro_event_list_ = sorted(retro_event_list_, key=lambda x: x['event_time'])
            # save
            retro_event_dict[pred_rel] = retro_event_list_

        seq_idx, pos_idx = eval(o_idx)
        real_event_ = source_data[int(seq_idx)][int(pos_idx)]
        real_event_sample, noise_event_sample, real_rel, noise_rel = make_complete_event_sequence(real_event_,
                                                                                                  retro_event_dict,
                                                                                                  pred_type=pred_type,
                                                                                                  data_name=data_name)

        res_i = (
            o_idx,
            real_event_sample,
            noise_event_sample,
            real_rel,
            noise_rel
        )

        if real_event_sample is not None:
            with DDB.at(ebm_db_name, pred_type).session() as (sess, ebm_db):
                ebm_db[str(o_idx)] = res_i
                sess.write()
            print('--- ok', o_idx)
        # return res_i

    if not DDB.at(ebm_db_name, pred_type).exists():
        DDB.at(ebm_db_name, pred_type).create()

    existed_ebm_db_dict = DDB.at(ebm_db_name, pred_type).read()

    dp_list = []
    gpt_db_dict = DDB.at(gpt_db_name).read()
    for pred_seq in pred_data:
        for pred_tuple in pred_seq:
            str_idx = '(\'' + str(pred_tuple['seq_idx']) + '\', \'' + str(pred_tuple['original_idx']) + '\')'
            thresh = topk - 1
            if str_idx in existed_ebm_db_dict and len(existed_ebm_db_dict[str_idx][2]) >= thresh:
                continue
            topk_pred = np.argsort(pred_tuple[f'pred_{pred_type}_score'])[-topk:]

            if str_idx not in gpt_db_dict:
                print(f'Idx {str_idx} is not in gpt_db')
                continue

            # just retrieve and add the label event
            topk_causal_dict = {}
            for pred in set([pred_tuple[f'label_{pred_type}']] + topk_pred.flatten().tolist()):
                if str(pred) not in gpt_db_dict[str_idx]:
                    print(f'-Miss {str_idx}-{pred} in gpt_db')
                    continue
                topk_causal_dict[str(pred)] = gpt_db_dict[str_idx][str(pred)]

            if len(topk_causal_dict) < topk:
                print(f'Causal events of the Idx {str_idx} are not fully fetched in gpt_db')
                continue

            res_i_prompt_event_dict = parse_events(topk_causal_dict, data_name=data_name)
            if len(res_i_prompt_event_dict) == 0:
                print(f'Idx {str_idx} can not be parse correctly')
                continue

            dp_list.append(
                (str_idx, res_i_prompt_event_dict)
            )

    with ProcessPool() as pool:
        pool.run(
            target=_process,
            dynamic_param_list=dp_list
        )

def make_comb_samples_for_energy_function(
        gpt_db_name,
        source_data,
        pred_data,
        ebm_db_name=None,
        retro_top_n=2
):
    import dictdatabase as DDB

    def _process(o_idx, p_e_d):
        try:
            if DDB.at(ebm_db_name, key=str(o_idx)).read() is not None:
                return

            retro_event_dict = {}
            # retrieve the close events
            total_time_list = get_total_time_list(source_data)

            for pred_rel, prompt_event_list in p_e_d.items():
                retro_event_list = [
                    retrieve_event_from_source_data(
                        target_event=prompt_event,
                        time_list=total_time_list,
                        source_data=source_data,
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

                retro_event_list_ = sorted(retro_event_list_, key=lambda x: x['time'])
                # save
                retro_event_dict[pred_rel] = retro_event_list_

            real_event_ = source_data[int(o_idx)]
            real_event_sample, noise_event_sample, real_rel, noise_rel = make_complete_event_sequence(
                real_event_,
                retro_event_dict,
                pred_type='rel_obj'
            )

            res_i = (
                o_idx,
                real_event_sample,
                noise_event_sample,
                real_rel,
                noise_rel
            )
            if real_event_sample is None:
                return

            with DDB.at(ebm_db_name).session() as (sess, ebm_db):
                ebm_db[str(o_idx)] = res_i
                sess.write()
            print('--- ok', o_idx)
        except Exception as e:
            print(e)

    if not DDB.at(ebm_db_name).exists():
        DDB.at(ebm_db_name).create()

    dp_list = []
    gpt_db_dict = DDB.at(gpt_db_name, '*').read()
    for pred_tuple in pred_data:
        str_idx = str(pred_tuple['original_idx'])

        if str_idx not in gpt_db_dict or len(gpt_db_dict[str_idx]) == 0:
            continue
        causal_events_dict = gpt_db_dict[str_idx]

        prompt_event_dict = parse_events(causal_events_dict)

        if len(prompt_event_dict) == 0:
            print(f'Idx {str_idx} can not be parse correctly')
            continue

        dp_list.append(
            (str_idx, prompt_event_dict)
        )

    with ProcessPool() as pool:
        pool.run(
            target=_process,
            dynamic_param_list=dp_list
        )