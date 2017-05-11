# Relation Classification via Multi-Level Attention CNNs

Code for the paper: [Relation Classification via Multi-Level Attention CNNs](http://iiis.tsinghua.edu.cn/~weblt/papers/relation-classification.pdf)


dataset: SemEval-2010 Task 8 Dataset

```
$ git clone git@github.com:FrankWork/acnn.git
$ cd acnn/
$ unzip data/embedding/senna/embeddings.zip data/embedding/senna/embeddings.txt
$ git remote -r
$ git checkout baseline
$ python main.py
Epoch: 1 Train: 11.53% Test: 30.11%
Epoch: 10 Train: 59.44% Test: 65.15%
Epoch: 50 Train: 89.76% Test: 75.81%

```

The performance is not so good as in the paper. Details in `log.txt`