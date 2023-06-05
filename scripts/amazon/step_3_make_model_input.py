import pickle
from datetime import datetime

import numpy as np
import pandas as pd


class InputProcessor:
    def __init__(self, source_csv_dir, train_end_date, valid_end_date):
        self.source_csv_dir = source_csv_dir
        self.source_df = pd.read_csv(source_csv_dir, header=0, encoding='latin-1')
        self.train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d')
        self.valid_end_date = datetime.strptime(valid_end_date, '%Y-%m-%d')

    def data_analysis(self, train_end_date, valid_end_date):
        self.source_df['review_time'] = pd.to_datetime(self.source_df['reviewTime'])

        train_df = self.source_df[self.source_df['review_time'] < train_end_date]
        print(f'num train data {len(train_df)}')

        valid_df = self.source_df[self.source_df['review_time'] < valid_end_date]
        print(f'num valid data {len(valid_df) - len(train_df)}')

        print(f'num test data {len(self.source_df) - len(valid_df)}')

        return

    def validate_user_df(self):
        user_list = np.unique(self.source_df['reviewerID'])
        print(f'num user {len(user_list)}')

        seq_len_list = []

        self.source_df['event_time'] = self.source_df['unixReviewTime'] / 86400

        res = pd.DataFrame()
        for idx, user in enumerate(user_list):
            user_df = self.source_df[self.source_df['reviewerID'] == user]
            user_df.sort_values(by=['unixReviewTime'], inplace=True)
            user_df.index = np.arange(len(user_df))
            user_df['event_dtime'] = user_df['event_time'].diff()
            user_df['review_time'] = pd.to_datetime(user_df['reviewTime'])
            seq_len_list.append(len(user_df))

            for row_idx in range(len(user_df)):
                if row_idx == 0:
                    user_df.loc[0, 'event_dtime'] = 0.0
                    prev_event_time = user_df.loc[0, 'event_time']
                    continue

                # if reviews at the same day, we set it as a random value + 3 hours interval
                if user_df.loc[row_idx, 'event_dtime'] == 0:
                    delta_t = 0.1 + np.random.uniform(-1, 1) * 0.05
                    user_df.loc[row_idx, 'event_dtime'] = delta_t
                    # need to update event_time
                    user_df.loc[row_idx, 'event_time'] = prev_event_time + delta_t
                else:
                    user_df.loc[row_idx, 'event_dtime'] = user_df.loc[row_idx, 'event_time'] - user_df.loc[
                        row_idx - 1, 'event_time']

                prev_event_time = user_df.loc[row_idx, 'event_time']

            # check
            diff_event_time = user_df['event_time'].diff().dropna()
            event_dtime = user_df['event_dtime'].values[1:]
            assert sum(abs(diff_event_time - event_dtime)) < 0.0001

            res = pd.concat([res, user_df])

        print(f'avg seq len {np.mean(seq_len_list)}')
        print(f'max seq len {np.max(seq_len_list)}')
        print(f'min seq len {np.min(seq_len_list)}')


        min_event_time = np.min(res['event_time'])
        res['event_time'] -= min_event_time

        return res

    def retrieve_text(self, input_text, num_tokens):
        if pd.isnull(input_text):
            return ''
        else:
            return input_text.replace('\n', ' ').replace('\r', '')[:num_tokens]

    def make_seq(self, total_user_df):
        user_list = np.unique(total_user_df['reviewerID'])
        res_seqs = {}
        for idx, user in enumerate(user_list):
            user_df = total_user_df[total_user_df['reviewerID'] == user]
            user_df.sort_values(by=['event_time'], inplace=True)
            user_df.index = np.arange(len(user_df))
            user_seq = []
            for i in range(len(user_df)):
                cate_text = self.retrieve_text(user_df.loc[i, 'cate_text_clean'], 50)
                title_text = self.retrieve_text(user_df.loc[i, 'title_text'], 50)
                summary_text = self.retrieve_text(user_df.loc[i, 'summary_text'], 50)
                review_text = self.retrieve_text(user_df.loc[i, 'review_text'], 1000)
                event_date = user_df.loc[i, 'reviewTime']
                event_date = datetime.strptime(event_date, '%m %d, %Y')
                event_date = event_date.strftime('%Y-%m-%d')

                event_text = 'product category:' + cate_text + ';product title:' + title_text + ';event time:' \
                             + event_date + ';summary text:' + summary_text + ';review text:' + review_text + '.'

                temp_dict = {'event_date': event_date,
                             'event_time': user_df.loc[i, 'event_time'],
                             'event_dtime': user_df.loc[i, 'event_dtime'],
                             'event_type': user_df.loc[i, 'cate_id'],
                             'event_text': event_text}
                user_seq.append(temp_dict)

            res_seqs[idx] = user_seq

        return res_seqs

    def save_to_pkl(self):

        total_df = self.validate_user_df()
        total_seqs = self.make_seq(total_df)

        with open('amazon_v0327.pkl', "wb") as f_out:
            pickle.dump(
                {
                    "dim_process": 24,
                    'user_seqs': total_seqs
                }, f_out
            )

        return


if __name__ == '__main__':
    source_dir = 'filer_user_v0308.csv'
    processor = InputProcessor(source_dir, train_end_date='2015-08-01', valid_end_date='2016-02-01')
    processor.data_analysis(train_end_date='2015-08-01', valid_end_date='2016-02-01')
    # # 46955, 6597, 12994  '2015-08-01', '2016-02-01'
    # # 49302, 5283, 11961
    # # 44561, 7844, 14141
    # processor.save_to_pkl()

    # entity_map = pd.read_csv('cate_type.csv', header=0)
    # res = entity_map[['cate_id', 'cate_text_clean']].loc[:24, :]
    # res = dict(res.values)
    #
    # with open('amazon_type_map.pkl', 'wb') as handle:
    #     pickle.dump(res, handle)
