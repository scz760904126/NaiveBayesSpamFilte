#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_
import SimpleNavieBayes.NavieBayes as naiveBayes
import random
import numpy as np

# 通过交叉验证测试分类的错误率
def testClassifyErrorRate():
    # 数据集预处理与存储
    filename = '../emails/training/SMSCollection.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    # 从训练集中随机选取测试集并从训练集中删除
    testWords = []
    testWordsType = []
    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        # 从训练集中删除要测试的数据
        del (smsWords[randomIndex])
        del (classLables[randomIndex])

    # 创建词库
    vocabularyList = naiveBayes.createVocabularyList(smsWords)
    # 构建词向量
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    trainMarkedWords = np.array(trainMarkedWords)
    # 通过词库和词向量计算P(S)、P(Wi|S) 、P(Wi|H)
    pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    # 计算联合概率进行分类
    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy, pSpam, testWords[i])
        print('预测类别：', smsType, '实际类别：', testWordsType[i])
        if smsType != testWordsType[i]:
            errorCount += 1

    print('Accuracy：', 1 - errorCount / testCount)


if __name__ == '__main__':
    testClassifyErrorRate()
