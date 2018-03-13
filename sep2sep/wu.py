import pandas as pd
import numpy as np
import jieba
kouchifile=pd.read_csv('kouchi.csv')
#print(kouchifile.head())
question = kouchifile['title'].astype('str')
answer=kouchifile['zhenduan'].astype('str')

cw = lambda x: ''.join(jieba.cut(x))
kouchifile['question'] = question.apply(cw)
kouchifile['words'] = answer.apply(cw)
X_question=kouchifile['question']
X_answer=kouchifile['words']


ignoring_words = ['【', '】','.',' ','。' ,'！','？','，','：',',','\xa0','?','0','1','2','3','4','5','6']

def preprocessing(sen):
    res = []
    for word in sen:
        if word not in ignoring_words:
            res.append(word)
    return res
question1 = [preprocessing(x) for x in X_question]
answer1 = [preprocessing(x) for x in X_answer]
# print(question1[:10])
# print(answer1[:2])
with open('question.csv','w') as f1:
    for line in question1:
        # print(line)
        f1.write(' '.join(line) + '\n')
# with open('answer.csv','w') as f2:
#     for line in answer1:
#         f2.write(list(line) +'/n')
