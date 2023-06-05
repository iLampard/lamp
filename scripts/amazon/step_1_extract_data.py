import gzip
import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l.decode("utf-8").replace('true', 'True').replace('false', 'False'))


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        if i < 5000000:
            i += 1
            continue
        df[i] = d
        i += 1
        if i % 100000 == 0:
            print(i)
        if i > 10000000:
            break

    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    review_df = pd.read_csv('review_df.csv', header=0)
    # meta_df = getDF(meta_dir)
    meta_df = pd.read_csv('meta_df.csv', header=0)
    meta_cols = ['category', 'description', 'brand', 'title', 'asin']
    meta_df = meta_df[meta_cols]
    res = pd.merge(review_df, meta_df[meta_cols], on='asin')
    res.to_csv('merge_df_v2.csv', header=True, index=False)
