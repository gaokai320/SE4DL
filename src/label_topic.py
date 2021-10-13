import os
import re
import sys
import csv
import random
from collections import defaultdict

import langid
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from markdown import markdown


model_dir = sys.argv[1]
topic_num = int(sys.argv[2])
iteration = int(sys.argv[3])

corpus_type = os.path.abspath(model_dir).split(os.path.sep)[-1].split('_')[0]
doc_topic_path = os.path.join(model_dir, '%s_%sdoctopics.txt' % (topic_num, iteration))
topic_keys_path = os.path.join(model_dir, '%x_%stopickeys.txt' % (topic_num, iteration))
doc_topic = {}
topic_docs = defaultdict(list)

with open(doc_topic_path) as f:
    for line in f:
        items = line.strip('\n').split("\t")
        doc = int(items[1])
        probs = [float(p) for p in items[2:]]
        topic = np.argmax(probs)
        doc_topic[doc] = (topic, probs[topic])
        topic_docs[topic].append((doc, probs[topic]))

with open(os.path.join("../result", '%s_%s_%s_doc_topic' % (corpus_type, topic_num, iteration)), 'w') as f:
    for d, t in doc_topic.items():
        f.write(str(d) + ',' + str(t[0]) + ',' + str(t[1]) + '\n')
for k, v in topic_docs.items():
    print(k, len(v))

names = ('Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
            'OwnerUserId', 'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate',
            'LastActivityDate', 'CommunityOwnedDate', 'ClosedDate', 'AnswerCount', 'CommentCount',
            'FavoriteCount', 'Tags', 'Title', 'Body')
posts_df = pd.read_csv("../data/posts.csv", names=names, index_col=0)
questions_df = posts_df[posts_df.PostTypeId == 1]
readme_list = [line.split(',')[0] for line in open("../data/readme_preprocessed")]

def remove_code(content):
    pattern = re.compile(r'```[\s\S]*?```')
    filter_content = re.sub(pattern, '', content)
    return filter_content

def extract_header_data(html):
    regex_code_remover= r'<code>.*?</code>'
    html=re.sub(regex_code_remover,'',html, flags=re.S)
    soup = BeautifulSoup(html,"html.parser")
    all_elements=soup.findAll()
    counth=len(soup.findAll('h1'))+len(soup.findAll('h2'))+len(soup.findAll('h3'))+len(soup.findAll('h4'))+len(soup.findAll('h5'))+len(soup.findAll('h6'))
    header_dict=defaultdict(str)
    cur_header = 'h100'
    if(counth>0):
        for e in all_elements:
            if(e.name in ['h1','h2','h3','h4','h5','h6']):
                cur_header=e.text
                header_dict[cur_header]=''
            else:
                if(e.text.strip('\n') not in header_dict[cur_header]):
                    header_dict[cur_header]= header_dict[cur_header]+" "+e.text.strip()
    else:
        header_dict['no_header']=soup.get_text()
    return dict(header_dict)

except_headers = ['reference', 'references', 'license', 'acknowledgement', 'author', 'requirements', 'authors', 'contributor', 'contributors', 'acknowledgements', 'licence','install', 'how to install', 'installation', 'table of contents', 'toc', 'usage', 'how to use']
def get_first_two_header(header_dict):
    cnt = 0
    content = ''
    for k, v in header_dict.items():
        if cnt < 2 and k.lower() not in except_headers:
            tmp = v
            if k not in ['h100', 'no_header']:
                tmp = k + ' ' + v
            if langid.classify(tmp)[0] == 'en':
                content = content + tmp
            cnt = cnt + 1
    return content

def get_question_by_idx(idx):
    return questions_df.iloc[idx].name, questions_df.iloc[idx].Title + questions_df.iloc[idx].Body

def get_readme_by_idx(idx):
    # print(idx)
    name = readme_list[idx]
    content = open("../data/readmes/" + name, encoding='utf-8', errors='ignore').read()
    content = remove_code(content)
    try:
        html = markdown(content)
        headerDict = extract_header_data(html)
        content = get_first_two_header(headerDict)
    except:
        content = name
    content = content.replace("\n", '.')
    return name, content


f = open('../result/label_%s.csv' % (corpus_type), 'w')
writer = csv.writer(f)
writer.writerow(['topic', 'id', 'content'])

if corpus_type == 'post':
    for topic, doc_probs in topic_docs.items():
        random_sample = random.sample(doc_probs, 30)
        for doc, prob in random_sample:
            name, content = get_question_by_idx(doc)
            writer.writerow([topic, name, content])

elif corpus_type == 'readme':
    for topic, doc_probs in topic_docs.items():
        random_sample = random.sample(doc_probs, 30)
        for doc, prob in random_sample:
            name, content = get_readme_by_idx(doc)
            writer.writerow([topic, name, content])

f.close()
