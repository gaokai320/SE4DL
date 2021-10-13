# On the Variability of Software Engineering Needs for Deep Learning: Stages, Trends, and Application Types
Wide use of Deep Learning (DL) has not been followed by the corresponding advances in software engineering (SE) for DL. Research shows that developers writing DL software have specific development stages (i.e., SE4DL stages) and face new DL-specific problems. Despite substantial research, it is not clear how such needs vary over stages, DL application types, or if they change over time. To help focus research and development efforts on DL-development challenges, we analyze 92,830 Stack Overflow (SO) questions and 227,756 READMEs of public repositories related to DL. Latent Dirichlet Allocation (LDA) reveals 27 topics for the SO questions with 19 (70.4%) question topics primarily relating to a single SE4DL stage and eight topics spanning multiple stages. Most questions concern Data Preparation and Model Setup stages. The relative rate of questions for 11 topics have increased, for nine topics decreased over time. Questions for the former 11 topics had a lower percentage of having an accepted answer than for the remaining topics. LDA reveals 26 themes for 227k repository README files. To group SO questions by application types, we apply LDA model fitted on READMEs to the 92,830 SO questions and find that 27% of the questions are related to 16 themes corresponding to distinct DL application types. The most common question topics for questions related to the application types include ten single-stage topics, with three topics primarily relating to Data Preparation and another three to Model Setup stage. Based on our findings, we distill several actionable insights for SE4DL research, practice, and education such as better support on operating trained models, improvement of DL framework compatibility, and application-type specific tools and teaching materials.

## Instructions

### Extract READMEs and SO questions

```shell
# ssh to WoC's da4 server to produce readmes.tar.gz
./extract_repos.sh
# transfer readmes.tar.gz from da4 server to data folder
scp da4:/data/play/dl-repos/dl_project ./data/
scp da4:/data/play/dl-repos/readmes.tar.gz ./data/
tar -zxvf readmes.tar.gz

# extract SO questions to produce posts.csv
wget -P ./data https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
cd data
7z x stackoverflow.com-Posts.7z
cd ../src
python extract_post.py ../data/Posts.xml
rm Posts.xml
```

The data files can be accessed at: https://drive.google.com/drive/folders/1BLbLgtyoFa4nX54gqaQO5HWEgV9G9mbJ?usp=sharing

### Preprocess READMEs and SO questions

```shell
cd src
python preprocess_data.py
```

### Run LDA on READMEs, questions, and combined READMEs and Questions
```shell
python -u tune_lda.py ~/mallet-2.0.8/bin/mallet 80 ../post_model ../data post 10 10 500 2000 5 50 100 100 2>../post_model/post_log
python -u tune_lda.py ~/mallet-2.0.8/bin/mallet 80 ../readme_model ../data readme 10 10 500 2000 5 50 100 100 2>../readme_model/readme_log
```

For SO question corpus, *K* = 27, *I* = 772 achieves the highest C_v score with 0.6245652526425263
For README corpus, *K* = 26, *I* = 891 achieves the highest C_v score with 0.5896546056936485

### Sample documents to label question topics, README topics, and combined topics
```shell
python label_topic.py ../post_model 27 772
python label_topic.py ../readme_model 26 891
```

### Analyze topic trend in SO questions
```shell
python analyze_trend.py
```

### Inference topics in SO questions with LDA model trained on README corpus
```shell
bin/mallet import-file --preserve-case --keep-sequence --remove-stopwords --token-regex "\S+" --input ../readme_model/post2readme.txt --output ../RQ3_results/post2readme.mallet --use-pipe-from ../readme_model/26_891corpus.mallet
bin/mallet infer-topics --input ../RQ3_results/post2readme.mallet --inferencer ../readme_model/26_891inferencer.mallet --output-doc-topics ../RQ3_results/post2readme_topics.infer --num-iterations 891 --doc-topics-threshold 0.0 --random-seed 10
```

### Merge question topic and README themes of SO questions
```shell
python inference.py
python merge_topics.py
```

### Run Raw Score

```shell
python -u run_raw_score.py ~/mallet-2.0.8/bin/mallet 80 ../post_raw_score ../data post 10 27 772 2>../post_raw_score/post_log
python -u run_raw_score.py ~/mallet-2.0.8/bin/mallet 80 ../readme_raw_score ../data readme 10 26 891 2>../readme_raw_score/post_log
```

## Requirements

```
Python3.8.3
gensim==3.8.3
pandas==1.0.5
langid=1.1.6
beautifulsoup4==4.9.1
markdown==3.3.3
nltk==3.5
tqdm==4.47.0
pyevolve==0.6 https://github.com/BubaVV/Pyevolve
```

