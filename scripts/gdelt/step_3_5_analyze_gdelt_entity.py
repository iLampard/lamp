import collections
import pickle

import numpy as np
import pandas as pd



def get_event_given_entity_id(source_data, entity_id, id_pos=1):
    events =  filter(lambda x: x[id_pos] == entity_id, source_data)
    return events



def main():
    # load all events
    data_dir = '../..data/gdelt/gdelt.pkl'
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)

    data = data['data']

    entity_map_csv_dir = '../../scripts/entity_map.csv'
    entity_map = pd.read_csv(entity_map_csv_dir, header=0)
    entity_map.rename(columns={'index': 'obj_id'}, inplace=True)

    # an overall statistics for the object
    total_obj = [x[1] for x in data]
    counter = collections.Counter(total_obj)
    top_20_obj = dict(counter.most_common(20))

    top_20_obj_list = top_20_obj.keys()
    obj_pct = np.array(list(top_20_obj.values())) / len(total_obj)

    obj_stat = pd.DataFrame(columns=['obj_id', 'obj_pct'])
    obj_stat['obj_id'] = top_20_obj_list
    obj_stat['obj_pct'] = obj_pct
    obj_stat = pd.merge(obj_stat, entity_map, on='obj_id', how='left')
    obj_stat = obj_stat[['obj_id', 'obj_pct', 'actor_code']]

    # a detailed look at every event
    total_sub = [x[0] for x in data]
    obj_spec = {}
    for sub_id in total_sub:
        # top 20 sub given the obj
        all_events_given_obj_id = get_event_given_entity_id(data, obj_id)
        all_suj_given_obj_id = [x[0] for x in all_events_given_obj_id]

        top_20_sub_given_obj_id = collections.Counter(all_suj_given_obj_id).most_common(20)
        top_20_sub_given_obj_id = list(top_20_sub_given_obj_id.values())

        # top 20 obj given obj's top sub
        top_20_obj_one_step = []
        for sub_id in top_20_sub_given_obj_id:
            all_events_given_sub_id = get_event_given_entity_id(data, sub_id, id_pos=0)
            all_obj_given_sub_id = [x[1] for x in all_events_given_sub_id]
            top_20_obj_given_sub_id = collections.Counter(all_obj_given_sub_id).most_common(5)
            top_20_obj_one_step.extend(list(top_20_obj_given_sub_id.values()))


    res = {'obj_stat': obj_stat,
           'obj_spec': obj_spec}

    with open('gdelt_obj_stat_v0327.pkl', "wb") as f_out:
        pickle.dump(res, f_out)

    return


if __name__ == '__main__':
    main()