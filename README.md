# 1.4亿通用知识图谱问答平台开发教程

本文基于ownthink 1.4亿条三元组，共4572万实体，1.41亿实体关系，从零开始搭建一种通用知识图谱问答平台，详细目录如下：

* 数据下载及预处理
* neo4j知识图谱构建
* 图谱问答预测流程介绍
* 生成训练样本
* 训练模型
* 相似度匹配
* Flask后端发布
* Vue前端开发
* 特别鸣谢

> 如果github无法正常显示图片文件，参考[该文档](https://zhuanlan.zhihu.com/p/107196957)请自行在hosts文件下面添加下面内容
```
# GitHub Start 
192.30.253.112    Build software better, together 
192.30.253.119    gist.github.com
151.101.184.133    assets-cdn.github.com
151.101.184.133    raw.githubusercontent.com
151.101.184.133    gist.githubusercontent.com
151.101.184.133    cloud.githubusercontent.com
151.101.184.133    camo.githubusercontent.com
151.101.184.133    avatars0.githubusercontent.com
151.101.184.133    avatars1.githubusercontent.com
151.101.184.133    avatars2.githubusercontent.com
151.101.184.133    avatars3.githubusercontent.com
151.101.184.133    avatars4.githubusercontent.com
151.101.184.133    avatars5.githubusercontent.com
151.101.184.133    avatars6.githubusercontent.com
151.101.184.133    avatars7.githubusercontent.com
151.101.184.133    avatars8.githubusercontent.com
 # GitHub End
```

## （一）数据集下载及预处理

### 1.1 数据下载
1.4亿三元组数据集来自ownthink开源的中文知识图谱，点击[这里](https://github.com/ownthink/KnowledgeGraphData)下载，数据是以（实体、属性、值），（实体、关系、实体）混合的形式组织，数据格式采用csv格式, 打开详情如下：
```
$ head ownthink_v2.csv
实体,属性,值
胶饴,描述,别名: 饴糖、畅糖、畅、软糖。
词条,描述,词条（拼音：cí tiáo）也叫词目，是辞书学用语，指收列的词语及其释文。
词条,标签,文化
红色食品,描述,红色食品是指食品为红色、橙红色或棕红色的食品。
红色食品,中文名,红色食品
红色食品,是否含防腐剂,否
红色食品,主要食用功效,预防感冒，缓解疲劳
红色食品,适宜人群,全部人群
红色食品,用途,增强表皮细胞再生和防止皮肤衰老
```
使用python进行读取测试：
```
import sys
import csv

with open('ownthink_v2.csv', 'r', encoding='utf8') as fin:
  reader = csv.reader(fin)
  for index, read in enumerate(reader):
    print(read)
    
    if index > 10:
      sys.exit(0)
```
运行以上脚本输出结果：
```
['实体', '属性', '值']
['胶饴', '描述', '别名: 饴糖、畅糖、畅、软糖。']
['词条', '描述', '词条（拼音：cí tiáo）也叫词目，是辞书学用语，指收列的词语及其释文。']
['词条', '标签', '文化']
['红色食品', '描述', '红色食品是指食品为红色、橙红色或棕红色的食品。']
['红色食品', '中文名', '红色食品']
['红色食品', '是否含防腐剂', '否']
['红色食品', '主要食用功效', '预防感冒，缓解疲劳']
['红色食品', '适宜人群', '全部人群']
['红色食品', '用途', '增强表皮细胞再生和防止皮肤衰老']
['红色食品', '标签', '非科学']
['红色食品', '标签', '生活']
```

### 1.2 实体统计
统计数据集中实体的个数，并用词云展示，对实体大致有个概念
```
import collections
import os
import csv
​
dir_path = os.getcwd()
all_entity_file = open(dir_path + '/data/record_data/entity/allEntitySort.txt', 'a', encoding='utf-8')
# 使用依赖库collections自带统计函数
with open(dir_path + '/data/ownthink_v2.csv', 'r', encoding='utf-8') as ownfile:
    reader = csv.reader((line.replace('\0', '') for line in ownfile))
    entity_col = [row[0] for row in reader if len(row) == 3]
    word_counts = collections.Counter(entity_col)
    allEntity_counts_dict = dict(word_counts.most_common())
    for keys, index in allEntity_counts_dict.items():
        temp_str = keys + ":" + str(index) + "\n"
        all_entity_file.write(temp_str)
    all_entity_file.close
​
# 附本人自写词频统计函数
class WordCount(object):
    def listWordCount(wordlist:list):
        ans = dict()
        wordlist_len = len(wordlist)
        for i in range(wordlist_len):
            if wordlist[i] not in ans:
                ans[wordlist[i]] = 1
            else:
                ans[wordlist[i]] = ans[wordlist[i]] + 1
        # 排序
        sim_dict_order = sorted(ans.items(),key=lambda x: x[1],reverse=True)
        res = dict(sim_dict_order)
        return res
```

### 1.3 属性统计
统计三元组中的属性词频，并用词云展示，看看哪些高频属性是否可以合并
```
import csv
import os
import re
import collections
# import numpy as np
# from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
​
dir_path = os.getcwd()
​
# black_mask = np.array(Image.open(dir_path + '/data_process/black_mask.png'))
with open(dir_path + '/data/ownthink_v2.csv', 'r', encoding='utf-8') as ownfile:
    reader = csv.reader((line.replace('\0', '') for line in ownfile))
    print('预处理结束')
    prop_col = [(re.sub(r'[;:：；、,，.。%【】•、`？*①②③④◇●◆/（& --•·）\\()“”〗""》〖《\s\d]', '', row[1])).replace('\u200b', '').replace("[", '').replace("]", '')
                for row in reader if len(row) == 3]
    print('属性列表已生成，开始统计词频并生成top10000词云')
    word_counts = collections.Counter(prop_col)
    prop_word_counts_top10000 = dict(word_counts.most_common(10000))
    wc = WordCloud(
        font_path=dir_path + '/fonts/STXINGKA.TTF',  # 字体路劲
        scale=12,
        # collocations=False,
        background_color='white',  # 背景颜色
        width=1000,
        height=600,
        max_font_size=50,  # 字体大小
        min_font_size=10,
        # mask=black_mask,  #背景图片
        max_words=1000
    )
    wc.generate_from_frequencies(prop_word_counts_top10000)
    wc.to_file(dir_path + '/data/record_image/propCiyunPicture.png')
```

![属性词云](https://www.zq-ai.com/static/img/kgqa/wordcloud_props.png)




