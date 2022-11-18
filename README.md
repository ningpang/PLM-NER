# 命名实体识别（NER）
该项目主要用于非嵌套命名实体识别（NER），模型包括LSTM，BERT，BERT+LSTM。样例数据集有中文和英文两种。代码会继续完善，不接受监督。

## 主要环境依赖
- pytorch
- numpy
## 数据集介绍
- renmin（中文）
- conll2003（英文）

数据集放于dataset文件夹下，按数据集名称区分。

## 语言模型
- BERT-base-uncased
- BERT-base-chinese

目前只测试了BERT处理，同理可替换成roberta或者其他语言模型，在数据处理部分可能需要修改。用不同的语言模型可以通过bert_path参数修改。

## 代码使用

### 1. 训练

```
python train.py --bert_path bert-base-uncased --dataset conll2003 --model Bert
```

### 2. 测试
```
python test.py --bert_path ...
```
训练测试的具体参数可以在主文件或者argument.py中设置添加。


