import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import numpy as np


class CFS(object):

    def __init__(self, correlation_method='pearson'):
        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(
                'Correlation method must be pearson, spearman or kendall not {}!'.format(correlation_method))
        self.method = correlation_method
        self.corr = None

    def select_features(self, data_frame):
        self.corr = data_frame.corr(method=self.method)
        corr_target = abs(self.corr['next_weight'])
        # Get highly correlated features
        features = dict(corr_target[corr_target >= 2 / np.sqrt(len(data_frame))])
        features.pop('next_weight', None)

        del_features = []
        # Remove intercorrelated features
        for feat in features.keys():
            feat_corrs = abs(self.corr[feat])
            for f in features.keys():
                if f != feat and feat_corrs[f] > 0.5:
                    if features[feat] < features[f]:
                        del_features.append(feat)
                        break
                    else:
                        del_features.append(f)
        for f in set(del_features):
            features.pop(f)
        return data_frame[features.keys()]

    def heatmap(self):
        if self.corr is not None:
            ax = sns.heatmap(self.corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                             square=True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            # plt.savefig('heatmap.pdf', bbox_inches="tight")
            plt.show()
        else:
            raise UserWarning('Correlation matrix is not created yet, run select_features() first!')


def load_data():
    directory = '/path/to/data/'

    df = pd.DataFrame(pd.read_csv(directory))
    weight = list(df['weight'])
    next_day = []
    for i, w in enumerate(weight):
        if i == 0:
            continue
        elif i == len(weight) - 1:
            next_day.append(0)
        next_day.append(w)
    df['next_weight'] = next_day
    for sub_dir in os.listdir(directory):
        if os.path.isfile(directory + sub_dir) or sub_dir.startswith("."):
            continue
        for file in os.listdir(directory + sub_dir):
            if file.endswith(".csv"):
                df_new = pd.DataFrame(pd.read_csv(directory + sub_dir + "/" + file))
                weight = list(df_new['weight'])
                next_day = []
                for i, w in enumerate(weight):
                    if i == 0:
                        continue
                    elif i == len(weight) - 1:
                        next_day.append(0)
                    next_day.append(w)
                df_new['next_weight'] = next_day
                df = df.append(pd.DataFrame(df_new), ignore_index=True)

    df.drop('datetime', axis=1, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    rand = []
    for i in range(len(df)):
        rand.append(rnd.random())
    df['random'] = rand
    return df


cfs = CFS()
data = pd.DataFrame(cfs.select_features(load_data()))
print(data.keys())
cfs.heatmap()
