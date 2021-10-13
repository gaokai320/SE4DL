from __future__ import division

import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import norm

names = ('Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
          'OwnerUserId', 'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate',
          'LastActivityDate', 'CommunityOwnedDate', 'ClosedDate', 'AnswerCount', 'CommentCount',
          'FavoriteCount', 'Tags', 'Title', 'Body')
posts_df = pd.read_csv("../data/posts.csv", names=names, index_col=0)
question_df = posts_df[posts_df.PostTypeId == 1][['Score', 'ViewCount', 'AnswerCount', 'FavoriteCount', 'CreationDate']]
question_df['Month'] = question_df['CreationDate'].apply(lambda x: '-'.join(x.split('-')[:2]))
topic_list = []
with open("../RQ2_results/post_27_772_doc_topic") as f:
    for line in f:
        topic_list.append(line.split(',')[1])
question_df['Topic'] = topic_list
question_df.reset_index(inplace=True)
print(question_df.head())
summary = question_df.groupby(['Month', 'Topic'])
month_count = summary['Id'].count().unstack()
month_count.fillna(0, inplace=True)
print(month_count)
month_count.to_csv("../RQ2_results/post_topic_month_count.csv")
month_ratio = month_count.div(month_count.sum(axis=1), axis=0)
month_ratio.to_csv("../RQ2_results/post_topic_month_ratio.csv")


results = []
topic_name = {}
with open("../RQ2_results/post_27_772_topics.csv") as f:
    for line in f:
        Id, name = line.split(',')[:2]
        topic_name[Id] = name
f = open("../RQ2_results/post_27_772_topic_trend.csv", 'w')
writer = csv.writer(f)
writer.writerow(['Id', 'Topic', 'Trend', 'Hypothesis', 'p-value', 'z-score',"Sen's slope"])
df = pd.read_csv('../RQ2_results/post_topic_month_ratio.csv', index_col=0)
for (columnName, columnData) in df.iloc[9:].iteritems():
    # print(columnData.values)
    trend, h, p, z, tau, s, var_s, slope, intercept = mk.original_test(columnData.values, 0.05)
    results.append([columnName, topic_name[columnName], trend, h, p, z, slope])
results.sort(key=lambda x: int(x[0]))
for r in results:
    writer.writerow(r)
f.close()
