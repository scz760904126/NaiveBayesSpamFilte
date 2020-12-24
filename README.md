# NaiveBayesSpamFilte
## Requirements ##
- python 3.7.0 or later
- numpy 1.19.1 or later
## Usage ##
### emails ###
垃圾邮件数据集，下载地址（[http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)）
### SimpleNavieBayes ###
NavieBayes.py 定义数据加载，处理以及贝叶斯分类的函数 <br/>
training.py 进行训练 <br/>
test.py 交叉验证进行测试
### 运行代码 ###
    cd SimpleNavieBayes
    python training.py
	python test.py
