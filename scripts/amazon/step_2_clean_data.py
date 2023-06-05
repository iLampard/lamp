import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class AmazonDataProcessor:
    def __init__(self,
                 source_dir,
                 target_clean_dir='clean_v0223.csv',
                 target_user_filter_dir='filer_user_v0306.csv'):
        self.source_dir = source_dir
        self.target_clean_dir = target_clean_dir
        self.target_user_filter_dir = target_user_filter_dir

    @staticmethod
    def retrieve_description_text(x):
        text = eval(x)[0] if not pd.isnull(x) else x
        return text

    @staticmethod
    def retrieve_cate_text(x):
        text = eval(x)
        text = text[1:]
        if text[0] in ['Women', 'Men']:
            if len(text) > 1:  # ['Men' 'Shoes'] => 'Men Shoes'
                res = text[0] + ' ' + text[1]
            else:  # ['Man'] = > 'Men Others'
                res = text[0] + ' General'
        elif text[0] in ['Boys', 'Girls', 'Baby', 'Baby Girls', 'Baby Boys']:
            if len(text) > 1:  # ['Boys' 'Shoes'] => 'Men Shoes'
                if text[1] not in ['Baby', 'Baby Girls', 'Baby Boys']:
                    res = 'Children ' + text[1]
                elif len(text) > 2:
                    res = 'Children ' + text[2]
                else:
                    res = 'Children General'
            else:  # ['Man'] = > 'Boys Others'
                res = 'Children General'
        else:
            res = text[0]

        return res

    @staticmethod
    def cate_map(cate_text, cate_event_map_df, target_col):
        res = cate_event_map_df[cate_event_map_df['cate_text'] == cate_text][target_col].to_list()[0]
        return res

    def _1_clean_df(self):
        source_df = pd.read_csv(self.source_dir, header=0)
        print(len(source_df))
        keep_cols = ['reviewerID', 'asin', 'reviewTime', 'unixReviewTime', 'cate_id', 'cate_text',
                     'summary_text', 'review_text', 'title_text', 'description_text']

        source_df['cate_text'] = source_df['category'].apply(self.retrieve_cate_text)
        print(len(source_df))
        source_df['description_text'] = source_df['description'].apply(self.retrieve_description_text)

        print(len(source_df))

        source_df.rename(columns={'summary': 'summary_text', 'reviewText': 'review_text',
                                  'title': 'title_text'}, inplace=True)

        # pd.DataFrame(source_df['cate_text'].value_counts(normalize=True).reset_index()).to_csv('cate_type.csv',
        #                                                                                        header=True, index=False)


        # encode the cate as the event type
        le = LabelEncoder()
        source_df['cate_id'] = le.fit_transform(source_df['cate_text'])

        # to slow to do the encoding here, so i comment these lines
        # cate_map_df = pd.read_csv('cate_type.csv', header=0)
        # source_df['cate_id'] = source_df['cate_text'].apply(lambda x: self.cate_map(x, cate_map_df, target_col='cate_id'))
        # source_df['cate_text_clean'] = source_df['cate_text'].apply(lambda x: self.cate_map(x, cate_map_df, target_col='cate_text_clean'))
        # source_df.drop(columns=['cate_text'], inplace=True)
        source_df[keep_cols].to_csv(self.target_clean_dir, index=False, header=True)

        # 939254400
        print(min(source_df['unixReviewTime']))
        print(len(np.unique(source_df['cate_id'])))  # 436
        print(len(source_df))

        return

    def _2_filter_users(self, num_users=2500):
        merge_df = pd.read_csv(self.target_clean_dir, header=0)
        print(len(merge_df))

        res_1 = pd.DataFrame()
        idx = 0
        for name, group in merge_df.groupby(['reviewerID']):
            group.sort_values('unixReviewTime', inplace=True)
            # drop duplicate
            group.drop_duplicates(inplace=True, keep='first')

            # drop duplicates that have the same values on time, cate id , cate_text and summary text
            group.drop_duplicates(subset=['unixReviewTime',
                                          'cate_id',
                                          'cate_text',
                                          'summary_text',
                                          'review_text',
                                          'title_text',
                                          'description_text'], inplace=True, keep='first')

            if len(group) < 20:
                continue

            idx += 1
            print(idx)
            group.index = list(range(len(group)))

            # set some invalid cell to be blank
            for i in range(len(group)):
                if 'var aPageStart' in group.loc[i, 'title_text']:
                    group.loc[i, 'title_text'] = ''

            group['reviewerID'] = len(group) * [name]
            res_1 = pd.concat([res_1, pd.DataFrame(group)])
            if idx > num_users:
                break

        # remap the cate map
        # i do it here because it is much faster than do it for the whole dataset
        cate_map_df = pd.read_csv('cate_type.csv', header=0)
        res_1['cate_id'] = res_1['cate_text'].apply(lambda x: self.cate_map(x, cate_map_df, target_col='cate_id'))
        res_1['cate_text_clean'] = res_1['cate_text'].apply(
            lambda x: self.cate_map(x, cate_map_df, target_col='cate_text_clean'))

        res_1.to_csv(self.target_user_filter_dir, header=True, index=False)

        return


if __name__ == '__main__':
    # source_dir = '../../scripts/amazon/merge_df_v2.csv'
    # target_clean_dir = '../../scripts/amazon/clean_v0307.csv'
    # target_user_filter_dir = 'filer_user_v0308.csv'
    # data_processor = AmazonDataProcessor(source_dir=source_dir,
    #                                      target_clean_dir=target_clean_dir,
    #                                      target_user_filter_dir=target_user_filter_dir)
    # data_processor._2_filter_users()

    target_clean_dir = '../../scripts/amazon/clean_v0307.csv'
    source_df = pd.read_csv(target_clean_dir, header=0)
    # print(min(source_df['unixReviewTime']))   # 939254400
    # print(len(np.unique(source_df['cate_id'])))   # 155
    # print(len(source_df)) # 3641673 # 10296612
    target_id = 'A117Q3W4LPC9VL'
    source_df = source_df[source_df['reviewerID']==target_id]
    source_df.to_csv('tmp.csv',header=True, index=False)
    print(source_df)