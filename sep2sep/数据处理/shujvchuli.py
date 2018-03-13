#!/usr/bin/python
# -*- coding: utf-8 -*-
#把数据转化成折叠形式，以病情分析进行分割
import pandas as pd
import numpy as np
df=pd.read_csv('kouchi 1.csv')
# #print(df.head())
# # print(df['zhenduan'].str.strip().split('            ', expand=True).stack()[:7])
# # import pandas as pd
# zhenduan=df['zhenduan'].str.strip('zhenduan')
# m=df.drop('zhenduan',axis=1).join(df['zhenduan'].str.split('病情分析',expand=True).stack().reset_index(level=1, drop=True).rename('zhenduan'))
# print(df.head())
# m.to_csv('a.csv', sep=',', header=True, index=True)
#把数据按照指导意见进行分割
# import pandas as pd
# df=pd.read_csv('a.csv')
# n=df.drop('zhenduan',axis=1).join(df['zhenduan'].str.split('指导意见',expand=True).stack().reset_index(level=1, drop=True).rename('zhenduan'))
# print(df.head())
# n.to_csv('b.csv', sep=',', header=True, index=True)
# import pandas as pd
#去掉空值字符串
df=pd.read_csv('b.csv')
NONE_zhenduan = (df["zhenduan"].isnull()) | (df["zhenduan"].apply(lambda x: str(x).isspace()))
df_null = df[NONE_zhenduan]
df_not_null = df[~NONE_zhenduan]
# print(df_not_null)
df_not_null.to_csv('c.csv', sep=',', header=True, index=True)