import pandas as pd
import csv

post2readme_topic_map = [[0]*26 for i in range(27)]
outf = open('../result/post2readme_topics', 'w')
readme_topic = []
with open('../readme_model/post2readme_doc_topics.txt') as f:
    for line in f:
        res = []
        doc, topic_probs = line.split(',', maxsplit=1)
        doc = int(doc)
        for item in topic_probs[2:-2].split('), ('):
            topic, prob = item.split(', ')
            res.append((int(topic), float(prob[:-1])))
        res.sort(key=lambda x: x[1], reverse=True)
        outf.write(f'{doc},{res[0][0]},{res[0][1]}\n')
        readme_topic.append(res[0][0])
outf.close()

outf = open('../result/post_multi_topics', 'w')
writer = csv.writer(outf)
writer.writerow(['post_id', 'problems', 'software type'])
with open("../result/post_27_772_doc_topic") as f:
    for line in f:
        doc, topic = line.split(',')[:2]
        doc, topic = int(doc), int(topic)
        writer.writerow([doc, topic, readme_topic[doc]])
        post2readme_topic_map[topic][readme_topic[doc]] += 1

df = pd.DataFrame(post2readme_topic_map, columns=[line.split(',')[1] for line in open('../result/readme_26_891_topics.csv') if not line.startswith('Topic No')], index=[[line.strip('\n').split(',')[5] for line in open('../result/post_27_772_topics.csv') if not line.startswith('Topic No')],
    [line.split(',')[1] for line in open('../result/post_27_772_topics.csv') if not line.startswith('Topic No')]])
df.to_csv("../result/topic_map.csv")

outf.close()
