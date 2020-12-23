#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_
import numpy as np
import SimpleNavieBayes.NavieBayes as naiveBayes

filename = '../emails/training/SMSCollection.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)
vocabularyList = naiveBayes.createVocabularyList(smsWords)
trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
# 转成array向量
trainMarkedWords = np.array(trainMarkedWords)
pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
fpSpam = open('pSpam.txt', 'w')
spam = pSpam.__str__()
fpSpam.write(spam)
fpSpam.close()
# 保存训练生成的语料库信息
# 保存语料库词汇
fw = open('vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\t')
fw.flush()
fw.close()
# 保存训练阶段获取的参数：pWordsSpamicity和pWordsHealthy
np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')  # savetxt和上面那种组合open和write的方法有什么不同。
np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')
