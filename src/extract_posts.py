# extract posts whose tags contain pytorch, tensorflow or tf, keras
import re
import sys
from lxml import etree
from tqdm import tqdm
import csv
header = ('Id', 'PostTypeId', 'ParentId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
          'OwnerUserId', 'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate',
          'LastActivityDate', 'CommunityOwnedDate', 'ClosedDate', 'AnswerCount', 'CommentCount',
          'FavoriteCount', 'Tags', 'Title', 'Body')
parser = etree.iterparse(sys.argv[1])
posts = open('../data/posts.csv', 'w')
writer = csv.writer(posts)
questions = []
for event, element in tqdm(parser):
    if element.tag == 'row':
        post_type_id = str(element.attrib.get('PostTypeId', ''))
        if post_type_id == '1':
            tags = element.attrib.get('Tags', '')
            matchobj = re.findall(r'[<](.*?)[>]', tags)
            matchobj = [tag.lower() for tag in matchobj]
            if 'pytorch' in matchobj or 'tensorflow' in matchobj or 'keras' in matchobj:
                tags = ";".join(matchobj)
                questions.append(element.attrib.get('Id', ''))
                rowcontent = [element.attrib.get(attribute, '') for attribute in header]
                rowcontent[-3] = tags
                writer.writerow(rowcontent)
        elif post_type_id == '2':
            ParentId = element.attrib.get('ParentId', '')
            if ParentId in questions:
                rowcontent = [element.attrib.get(attribute, '') for attribute in header]
                writer.writerow(rowcontent)
posts.close()