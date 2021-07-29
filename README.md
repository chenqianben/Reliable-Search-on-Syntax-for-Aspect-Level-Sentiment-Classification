# Reliable-Search-on-Syntax-for-Aspect-Level-Sentiment-Classification (RSSG)

## One-Sentence Summary

RSSG effectively utilizes the dependency grammar to tackle the aspect-level sentiment classification task via performing an effective search on the syntax graph for learning a better sentence representation.

![network_structure](https://github.com/chenqianben/Reliable-Search-on-Syntax-for-Aspect-Level-Sentiment-Classification/blob/main/assets/network_structure.PNG)

## Abstract

Aspect-level sentiment classification (ALSC) aims to predict the sentiment polarity of each aspect in a given sentence. It is a more challenging information retrieval task as compared to the sentence-level sentiment classification task owing to the fact that it is a fine-grained task which requires a model to attend to specific opinionated words related to the aspect. Leveraging the relationship between the aspect terms and the context words is crucial for encoding the sentence in this very task. Recent works explore the dependency tree to model the syntactic relationship between the aspect and context words of the sentence to boost model performance. Despite the promising performance, some works convert the dependency tree into a syntactic structure in a way that destructs the structural information of the tree while others do not effectively use the structural information in the tree. As a solution, we first construct a syntax graph that preserves the structural information of the dependency tree. We then propose a reliable syntax-based neural network model that performs a thorough search on the syntax graph to effectively find the relevant context information with respect to the aspect term for the sentence encoding. Noting that dependency trees parsed from existing dependency parsers may contain incorrect syntactic dependencies due to grammatical errors in a sentence, we adopt a convolutional neural network that takes into account the relations among features of words in a local neighborhood to mitigate the issues brought by incorrect syntactic dependencies. Our results on standard benchmark datasets demonstrate that our model outperforms the previous methods and achieves state-of-the-art results for the ALSC task.

## Requirement

- Python 3.6
- PyTorch 1.2.0
- NumPy 1.17.2

## Preparation

### Create an anaconda environment [Optional]:

```bash
conda create -n rssg
conda activate rssg
pip install -r requirements.txt
```

### Prepare the training data:

- The dataset files are already in the `data` folder.
- The directory structure is as follows:
```
data
├── Restaurants_Trial.json
├── Restaurants_Train.json
├── Restaurants_Test.json
├── Laptops_Trial.json
├── Laptops_Train.json
├── Laptops_Test.json
├── Tweets_Train.json
├── Tweets_Test.json
├── Restaurants16_Trial.json
├── Restaurants16_Train.json
└── Restaurants16_Test.json
```

### Download the pretrained embeddings:

#### GloVe embeddings
- Download pre-trained GloVe word vectors [here](https://nlp.stanford.edu/projects/glove/).
- Extract the [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) to the `glove` folder.

#### Bert embeddings
- Download pre-trained Bert word vectors [here](https://huggingface.co/bert-base-uncased/tree/main).
- Extarct the bert_config.json、pytorch_model.bin and covab.text to the `pretrained_bert` folder.

## Usage

### Train the model:

```bash
python train.py --dataset [dataset]
```

### Show help message and exit:

```bash
python train.py -h
```

## File Specifications

- **model.py**: Description for the model architecture.
- **data_utils.py**: Used functions for data preprocessing.
- **layer.py**: Description for the LSTM layer.
- **loss_func.py**: The loss function for optimizing the models.
- **train.py**: The scripts for training and evaluating the models.

## Contact

qianbenchen@buaa.edu.cn
