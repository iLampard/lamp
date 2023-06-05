import datetime
import pickle
import time

import numpy as np
import pandas as pd


class InputProcessor:
    def __init__(self, source_csv_dir):
        self.source_csv_dir = source_csv_dir
        self.source_df = pd.read_csv(source_csv_dir, header=0)

        self.process_df = self.source_df.copy()
        self.entity_map_csv_dir = '../entity_map.csv'
        self.entity_map, self.num_entity, self.num_rel = self.data_analysis(entity_map_csv_dir=self.entity_map_csv_dir)

    def map_actor_code_to_text(self, rel_code):
        map_dict = {1: 'MAKE STATEMENT',
                    2: 'APPEAL',
                    3: 'EXPRESS INTENT TO COOPERATE',
                    4: 'CONSULT',
                    5: 'ENGAGE IN DIPLOMATIC COOPERATION',
                    6: 'ENGAGE IN MATERIAL COOPERATION',
                    7: 'PROVIDE AID',
                    8: 'YIELD',
                    9: 'INVESTIGATE',
                    10: 'DEMAND',
                    11: 'DISAPPROVE',
                    12: 'REJECT',
                    13: 'THREATEN',
                    14: 'PROTEST',
                    15: 'EXHIBIT MILITARY POSTURE',
                    16: 'REDUCE RELATIONS',
                    17: 'COERCE',
                    18: 'ASSAULT',
                    19: 'FIGHT',
                    20: 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE'}
        return map_dict[rel_code]

    def data_analysis(self, entity_map_csv_dir):
        # save a entity_map
        # merge subject and object
        total_entity = pd.concat([self.source_df['Actor1Code'], self.source_df['Actor2Code']])

        entity_map = total_entity.value_counts(normalize=True)
        entity_map = entity_map.reset_index()
        entity_map.columns = ['actor_code', 'ratio']
        num_entity = len(entity_map)
        entity_map['index'] = range(num_entity)

        num_rel = len(set(self.source_df['EventRootCode']))

        entity_map.to_csv(entity_map_csv_dir, index=False, header=True)

        return entity_map, num_entity, num_rel

    def encode_col(self, to_encode, encode_map_df, encode_map_df_col):
        res = encode_map_df[encode_map_df[encode_map_df_col] == to_encode]['index'].to_list()[0]
        return res

    def encode_entity(self):
        """

        Returns: encoded entity number

        """

        # self.entity_map = pd.read_csv(self.entity_map_csv_dir, header=0)

        self.process_df['encode_actor1'] = self.process_df['Actor1Code'].apply(
            lambda x: self.encode_col(x, self.entity_map, 'actor_code'))

        self.process_df['encode_actor2'] = self.process_df['Actor2Code'].apply(
            lambda x: self.encode_col(x, self.entity_map, 'actor_code'))

        return

    def encode_rel(self):
        """

        Returns: encoded relation = EventRootCode - 1

        """

        self.process_df['encode_rel'] = self.process_df['EventRootCode'] - 1

        return

    def encode_time(self, s):
        return time.mktime(datetime.datetime.strptime(str(s), "%Y%m%d%H%M%S").timetuple())

    def analyze_time(self):
        res = pd.DataFrame(columns=['encode_actor', 'avg_dt'])
        for i in range(20):
            entity_df = self.process_df[self.process_df['encode_actor1'] == i]
            entity_df.sort_values(['event_time_unix'], inplace=True)
            dt = entity_df['event_time_unix'].diff(1).dropna()
            res.loc[i, 'avg_dt'] = np.mean(dt[dt > 0])
            res.loc[i, 'encode_actor'] = i

        print(res)
        return

    def clean_timestamp(self):
        self.process_df['event_time_unix'] = self.process_df['event_time'].apply(lambda x: self.encode_time(x))
        min_timestamp = min(self.process_df['event_time_unix'])

        self.process_df['event_time_unix'] -= min_timestamp
        self.process_df['event_time_unix'] /= (60 * 60 * 12)  # half day

        self.process_df.sort_values(['event_time_unix'], inplace=True)
        self.process_df.reset_index(inplace=True)

        return

    def make_event_text(self, actor1_name, actor2_name, rel, event_time, event_headline, event_key_phrases):
        rel_ = self.map_actor_code_to_text(rel)

        if pd.isnull(event_headline):
            event_headline = ''

        event_time = str(event_time)[:8]
        event_time = event_time[:4] + '-' + event_time[4:6] + '-' + event_time[6:8]

        res = 'event headline:' + event_headline + '; event type:' + rel_ + '; event time:' + str(
            event_time) + '; subject name:' + actor1_name + '; object name:' + actor2_name +  '.'

        return res

    def make_seq(self, input_df):
        input_df['event_text'] = input_df.apply(lambda x: self.make_event_text(x['Actor1Name'],
                                                                               x['Actor2Name'],
                                                                               x['EventRootCode'],
                                                                               x['event_time'],
                                                                               x['news_title'],
                                                                               x['news_short_key_word']),
                                                axis=1)

        seq_tuples = zip(input_df['encode_actor1'],
                         input_df['encode_actor2'],
                         input_df['encode_rel'],
                         input_df['event_time_unix'],
                         input_df['event_text'])
        seq_entity = set(input_df['encode_actor1']).union(set(input_df['encode_actor2']))

        return list(seq_tuples), seq_entity

    def make_dataset(self, train_ratio=0.7, valid_ratio=0.1):
        self.encode_rel()
        self.encode_entity()
        self.clean_timestamp()

        current_seq, current_entity = self.make_seq(self.process_df)

        self.save_to_pkl(current_seq, self.num_entity, self.num_rel, '../gdelt.pkl')
        # total_len = len(current_seq)
        # train_seq = current_seq[: int(total_len * train_ratio)]
        # valid_seq = current_seq[int(total_len * train_ratio): int(total_len * (train_ratio + valid_ratio))]
        # test_seq = current_seq[int(total_len * (train_ratio + valid_ratio)):]
        #
        # self.save_to_pkl(train_seq, self.num_entity, self.num_rel, '../train.pkl')
        # self.save_to_pkl(valid_seq, self.num_entity, self.num_rel, '../dev.pkl')
        # self.save_to_pkl(test_seq, self.num_entity, self.num_rel, '../test.pkl')

        return

    def make_csv_data(self):
        self.encode_rel()
        self.encode_entity()
        self.clean_timestamp()

        process_df = self.process_df[['encode_actor1', 'encode_actor2', 'encode_rel', 'event_time_unix']]

        total_len = len(process_df)
        train_ratio = 0.7
        valid_ratio = 0.1
        train_df = process_df.loc[: int(total_len * train_ratio), :]
        test_df = process_df.loc[int(total_len * (train_ratio + valid_ratio)):, :]

        train_df.to_csv('gdelt_train.csv', header=True)
        test_df.to_csv('gdelt_test.csv', header=True)

        return

    def save_to_pkl(self, batch_data, num_entity, num_rel, save_dir):
        res = dict({'num_entity': num_entity,
                    'num_rel': num_rel,
                    'data': batch_data})
        with open(save_dir, 'wb') as handle:
            pickle.dump(res, handle)

        return


if __name__ == '__main__':
    csv_dir = '../../preprocess/gdelt_v0314_v2.csv'
    processor = InputProcessor(source_csv_dir=csv_dir)
    # processor.data_analysis('entity_map.csv')

    processor.make_dataset()
