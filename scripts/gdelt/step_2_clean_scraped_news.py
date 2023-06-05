import os

import pandas as pd
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

class NewsProcessor:
    def __init__(self, raw_gdelt_csv_dir, download_news_dir, clean_gdelt_csv_dir):
        # key word extractor
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(model=self.sentence_model)
        self.raw_gdelt_csv_dir = raw_gdelt_csv_dir
        self.clean_gdelt_csv_dir = clean_gdelt_csv_dir
        self.download_news_dir = download_news_dir

    @staticmethod
    def parse_txt(txt_dir):
        # pay attention to txt dir
        # 只能是类似 'data/gdelt_news/20220424164500/1040966575/article_Jaishankar likely to visit Dhaka April 28_lite'，
        # 不能有更多的 _
        news_name = txt_dir.split('.')[0]
        news_name = news_name.split('_')[2]

        news = open(txt_dir, 'r')
        news_content = news.read()
        news.close()

        return news_name, news_content

    @staticmethod
    def clean_news_txt(news_content):
        # 包含许多中文，其实是报错信息： 请检查您的互联网连接是否正常 请检查所有网线是否都已连好，然后重新启动您可能正在使用的任何路由器、调制解调器或其他网络设备。
        if '请检查您的互联网连接是否正常' in news_content:
            return False

        # too short, wrong text  # Discover new and used cars for sale near you Check out the updated Microsoft Start Autos today!
        # event id
        if len(news_content) < 100:
            return False

        # too short, less than 3 lines
        if len(news_content.splitlines()) <= 3:
            return False

        return True

    def get_article_keyword_dir(self, event_id_path):
        article_dir, keyword_dir = None, None
        dir_list = os.listdir(event_id_path)
        for dir_ in dir_list:
            if dir_.startswith('article'):
                article_dir = dir_
            elif dir_.startswith('keyword'):
                keyword_dir = dir_
        return article_dir, keyword_dir

    def merge_download_files_to_csv(self, db_storage, save_dir):
        """Merge all the downloaded news txt files, filter them, use kyeBert to extract short/long keywords,
        save it locally to txt files and save all to one csv files.

        The whole process may fail so i save every extracted keyword in txt format along with the article txt.

        top-3 key words are returned and concatenated to one string.
        short keyword: length 4 - 7 words
        """
        res = pd.DataFrame(columns=['GLOBALEVENTID', 'event_time',
                                    'news_title', 'news_short_key_word'])
        count = 0
        raw_count = 0

        for idx, ts_id in enumerate(os.listdir(db_storage)):
            ts_id_path = os.path.join(db_storage, ts_id)
            if os.path.isdir(ts_id_path):
                for event_id in os.listdir(ts_id_path):
                    event_id_path = os.path.join(ts_id_path, event_id)

                    if os.path.isdir(event_id_path) and len(os.listdir(event_id_path)) > 0:

                        raw_count += 1

                        if raw_count % 100 == 0:
                            print('raw_count ' + str(raw_count))

                        article_dir, keyword_dir = self.get_article_keyword_dir(event_id_path)

                        if keyword_dir is not None:
                            continue  # keyword already extracted, skip it

                        article_dir = os.path.join(event_id_path, article_dir)
                        news_title, news_content = NewsProcessor.parse_txt(article_dir)

                        add_content = NewsProcessor.clean_news_txt(news_content)

                        if add_content:
                            # short keyword
                            # keyphrase_ngram_range=(4, 7)
                            key_word = self.kw_model.extract_keywords(news_content,
                                                                      keyphrase_ngram_range=(4, 7),
                                                                      use_mmr=True,
                                                                      top_n=3)
                            key_word = list(zip(*key_word))[0]
                            key_word = ','.join(key_word)

                            res.loc[count, 'GLOBALEVENTID'] = event_id
                            res.loc[count, 'event_time'] = ts_id
                            res.loc[count, 'news_title'] = news_title
                            res.loc[count, 'news_short_key_word'] = key_word

                            count += 1

                    if count > 0 and count % 100 == 0:
                        print(count)

        res.to_csv(save_dir, index=False, header=True)
        return

    def read_gdelt_news_dir(self, csv_dir):
        res = pd.read_csv(csv_dir, header=0)
        return res

    def merge_with_raw_gdelt(self, news_csv_dir, raw_csv_dir, target_dir):
        """ Merge with raw gdelt file to add back other information
        """
        df_new = self.read_gdelt_news_dir(news_csv_dir)
        df_raw = self.read_gdelt_news_dir(raw_csv_dir)

        res = pd.merge(df_raw, df_new, on=['GLOBALEVENTID'], how='right')
        print(len(res))
        res.to_csv(target_dir, index=False, header=True)
        return

    def remove_invaliad_str(self, news_title):
        """ remove the string that is a mix of number and alphabet, e.g., 35c830700f9c

        because the headline parsed from url could be
        tutu-wrap-everything-you-need-to-know-about-the-archs-funeral-90b07d7e-3a72-433f-ab64-35c830700f9c
        """
        res = []
        for x in news_title.split(' '):
            if x.isnumeric() or x.isalpha():
                res.append(x)
        return ' '.join(res)

    def clean_news_headline(self, source_dir, target_dir):
        res = pd.read_csv(source_dir, header=0)

        for i in range(len(res)):
            # if res.loc[i, 'news_title'] in['Access denied', 404, '404', '404 Not Found', 'U', 'Article'] :
            if len(str(res.loc[i, 'news_title'])) < 20:
                url_split = res.loc[i, 'SOURCEURL'].split('/')
                # 有时候标题会在倒数第三个，比如这种
                # https://www.kob.com/health-news/revelers-await-return-to-nycs-times-square-to-usher-in-2022/6346232/?cat=600
                if len(url_split[-1]) > 10:
                    title = url_split[-1]
                elif len(url_split[-2]) > 10:
                    title = url_split[-2]
                else:
                    title = url_split[-3]

                # replace the '-' to space
                # u-s => us, meidi
                # u-n => un, united nation
                title = title.replace('u-s', 'us')
                title = title.replace('u-n', 'un')

                # 2021-marks-four-wasted-years-under-cyril-ramaphosa => 2021 marks four wasted years under cyril ramaphosa
                title = title.replace('-', ' ')

                # _xx_xxx => xx  xx
                title = title.replace('_', ' ')

                title = self.remove_invaliad_str(title)
                print(i, title)
                res.loc[i, 'news_title'] = title

        res.to_csv(target_dir, index=False, header=True)

        return

    def run(self):
        print('Extract keywords from downloaded news and merge them together')
        news_merge_dir = 'temp_news.csv'
        self.merge_download_files_to_csv(db_storage=self.download_news_dir,
                                         save_dir=news_merge_dir)

        print('Merge news with other event information into one file')
        news_gdelt_merge_dir = 'temp_news_gdelt_merge.csv'
        self.merge_with_raw_gdelt(news_csv_dir=news_merge_dir,
                                  raw_csv_dir=self.raw_gdelt_csv_dir,
                                  target_dir=news_gdelt_merge_dir)

        print('Merge news with other event information into one file')
        self.clean_news_headline(source_dir=news_gdelt_merge_dir,
                                 target_dir=self.clean_gdelt_csv_dir)
        return


if __name__ == '__main__':
    news_handler = NewsProcessor(raw_gdelt_csv_dir='../../preprocess/events_raw_clean.csv',  #
                                 download_news_dir='../../preprocess/data/gdelt_newsxm2',  # dir of downloaded news txt files
                                 clean_gdelt_csv_dir='events_clean_v0302.csv')  # target save dir
    news_handler.run()
