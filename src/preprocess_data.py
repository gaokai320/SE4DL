import os
import sys
import re
from collections import defaultdict

import gensim
import gensim.corpora as corpora
import langid
import pandas as pd
from bs4 import BeautifulSoup
from markdown import markdown
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer


'''
preprocess readmes
'''
# clean retrived technical map
cnt = 0
outf = open('../data/filtered_project', 'w')
with open('../data/dl_project', errors='ignore') as f:
    for line in tqdm(f):
        items = line.strip('\n').split(';')
        cmt, proj, ts, author, blob, suffix, lang, fname = items[:8]
        pkgs = ','.join([module.split('.')[0] for module in items[8:]])
        # remove projects developed by tensorflow, keras, or pytorch organization
        if proj.startswith('tensorflow_') or proj.startswith('keras-team_') or proj.startswith('pytorch_'):
            continue
        # remove projects whole name related to self-learning
        elif re.findall('udacity|assignment|course|homework|class|lesson|tutorial|syllabus|mooc|paper|thesis|dissertation', proj, re.I):
            continue
        # remove false import "" stands for jupyter notebook

        elif 'tensorflow' not in pkgs and 'keras' not in pkgs and 'torch' not in pkgs:
            continue

        # language bindings:
        # TensorFlow: Python JavaScript C++ Java C# Haskell Julia MATLAB R Ruby Rust Scala Go Swift
        # Keras: Python R
        # PyTorch: Python C++ Javadoc
        elif suffix not in ['PY', 'C', 'java', 'Cs', 'jl', 'R', 'rb', 'Rust', 'Scala', 'Go', 'ipy']:
            continue
        
        else:
            outf.write(line)
            cnt = cnt + 1
print(f'{cnt} lines remained after cleaning dl_project files')

# get cleaned projects with English readmes
proj_with_readmes = set(os.listdir("../data/readmes"))
all_projs = set()
with open('../data/filtered_project') as f:
    for line in f:
        all_projs.add(line.split(';')[1])
proj_with_readmes = list(proj_with_readmes.intersection(all_projs))
print(f'{len(all_projs)} dl projects, {len(proj_with_readmes)} readmes')
proj2readme = {}
for p in tqdm(proj_with_readmes):
    content = open('../data/readmes/' + p, encoding='utf-8', errors='ignore').read().lower()
    if langid.classify(content)[0] == 'en':
        proj2readme[p] = content
print(f'{len(proj2readme)} English readmes')
en_readmes = set(proj2readme.keys())
for fn in set(proj_with_readmes) - en_readmes:
    os.remove("../data/readmes/" + fn)


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

stopwords = [line.strip('\n') for line in open("stopwords.txt")]
stopwords.extend(['tensorflow', 'pytorch', 'keras'])
print(f'Total number of stopwords: {len(stopwords)}')

# get each readme's first two headers and preprocess them
proj2processed_readme = {}
projs = []
readme_list = []
for p, readme in tqdm(proj2readme.items()):
    readme = remove_code(readme)
    # if len(re.sub(r'[^a-zA-Z]', ' ', content).split(' ')) < 100:
    #     continue
    try:
        html = markdown(readme)
        headerDict = extract_header_data(html)
        content = get_first_two_header(headerDict)
        if content != '':
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            content = url_pattern.sub(' ', content)
            content = re.sub(r'[^a-zA-Z]', ' ', content)
            projs.append(p)
            readme_list.append([word for word in content.split() if word not in stopwords])
    except:
        print(p)
print(f'{len(projs)} readmes are successfully preprocessed!')

# make bigram models
phrases = gensim.models.phrases.Phrases(readme_list, min_count=5, threshold=100)
bigram = gensim.models.phrases.Phraser(phrases)
bigram.save("../data/readme_bigram.pkl")
readme_list = [bigram[post] for post in readme_list]
for i, readme in enumerate(tqdm(readme_list)):
    proj2processed_readme[projs[i]] = ' '.join(readme)

readme_df = pd.DataFrame(data=proj2processed_readme.items(), columns=['project', 'preprocessed_readmes'])
readme_df['preprocessed_readmes'] = readme_df['preprocessed_readmes'].apply(str)
readme_df.set_index("project", inplace=True)
readme_df.drop(["kazakovantony_newyoudeeplearning"], inplace=True)
filter_df = readme_df[~readme_df['preprocessed_readmes'].str.contains("udacity|assignment|course|homework|class|lesson|tutorial|syllabus|mooc|paper|thesis|dissertation|nanodegree", flags=re.IGNORECASE)]
filter_df = readme_df
print(f'{len(filter_df)} readmes left after excluding learning project')
filter_df['word_count'] = filter_df['preprocessed_readmes'].map(lambda x: len(x.split(' ')))
filter_df = filter_df[filter_df.word_count >= 5]
print(f'{len(filter_df)} readmes left after exluding short readmes')
stemmer = SnowballStemmer('english')
def stem(readme):
    return [stemmer.stem(word) for word in readme.split(' ')]
filter_df['stemmed_readme'] = filter_df['preprocessed_readmes'].map(stem)
with open('../data/readme_preprocessed', 'w') as f:
    for row in filter_df.iterrows():
        p = row[0]
        r = row[1]['stemmed_readme']
        f.write(p + ',' + ' '.join(r)+'\n')
documents = filter_df['stemmed_readme'].values
id2word = corpora.Dictionary(documents)
id2word.filter_extremes(no_below=100, no_above=0.5)
corpus = [id2word.doc2bow(doc) for doc in documents]
id2word.save('../data/readme_id2word.dict')
corpora.MmCorpus.serialize('../data/readme_corpus.mm', corpus)
print(f'Total number of words: {len(id2word)}')
print(f'Total number of documents: {len(corpus)}')
print("Readmes preprocess done!")

# load posts
names = ('Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
          'OwnerUserId', 'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate',
          'LastActivityDate', 'CommunityOwnedDate', 'ClosedDate', 'AnswerCount', 'CommentCount',
          'FavoriteCount', 'Tags', 'Title', 'Body')
posts_df = pd.read_csv("../data/posts.csv", names=names, index_col=0)
print(f'Total number of questions and answers: {len(posts_df)}')
questions_df = posts_df[posts_df.PostTypeId == 1]
print(f'Total number of questions: {len(questions_df)}')
question_with_acc_cnt = len(questions_df[questions_df.AcceptedAnswerId.notna()])
print(f'Total number of questions with accepted answer: {question_with_acc_cnt}')

# preprocess posts
stopwords = [line.strip('\n') for line in open("stopwords.txt")]
stopwords.extend(['tensorflow', 'pytorch', 'keras'])
print(f'Total number of stopwords: {len(stopwords)}')
stemmer = SnowballStemmer('english')
def preprocess(df):
    post_list = []
    id_list = []
    for i in tqdm(range(len(df))):
        id_list.append(df.iloc[i].name)
        post = df.iloc[i].Title + '\n' + df.iloc[i].Body
        post = re.sub(r"<code>.*?</code>", "", post, flags=re.S)
        post = re.sub(r"<blockquote>.*?</blockquote>", "", post, flags=re.S)
        raw_text = BeautifulSoup(post, 'lxml').text
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        raw_text = url_pattern.sub(' ', raw_text.lower())
        raw_text = re.sub(r'[^a-zA-Z]', ' ', raw_text)
        post_list.append([word for word in raw_text.split() if word not in stopwords])
    phrases = gensim.models.phrases.Phrases(post_list, min_count=5, threshold=100)
    bigram = gensim.models.phrases.Phraser(phrases)
    bigram.save("../data/post_bigram.pkl")
    post_list = [bigram[post] for post in post_list]
    stemmed_post_list = []
    for post in post_list:
        stemmed_post_list.append([stemmer.stem(word) for word in post])
    return stemmed_post_list, id_list

def build_corpus(post_list, id_list):
    # index = list(questions_df.index)
    with open("../data/post_preprocessed", 'w') as f:
        for i, post in enumerate(post_list):
            f.write(str(id_list[i]) + ',' + ' '.join(post))
            f.write('\n')
    id2word = corpora.Dictionary(post_list)
    corpus = [id2word.doc2bow(post) for post in post_list]
    id2word.save('../data/post_id2word.dict')
    corpora.MmCorpus.serialize('../data/post_corpus.mm', corpus)
    print(f'Total number of words: {len(id2word)}')
    print(f'Total number of documents: {len(corpus)}')

stemmed_post_list, id_list = preprocess(questions_df)
print('Preprocess Done!')
build_corpus(stemmed_post_list, id_list)
print('Corpus built!')
