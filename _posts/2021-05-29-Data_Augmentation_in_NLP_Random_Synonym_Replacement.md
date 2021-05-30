---
title: "Data Augmentation in NLP: Random Synonym Replacement"
layout: single
category: Deep learning
tag:
 - NLP
 - Data Augmentation
toc: true
toc_sticky: true
---
# Abstract
Hello everyone, my name is Austin.  

Today I want to introduce one of the NLP data augmentation methods named random synonym replacement.  

In human conversation or writing, we use different words to represent the same thing.  
![synonym example](/assets/images/deep_learning/2021-05-29-Data_Augmentation_in_NLP_Random_Synonym_Replacement_image1.png)  

Therefore, this method is to use different words to express the same thing when simulating human daily conversation or writing.  

In this method, the key point is to use the synonym to replace the random select word to prevent the neural network overfitting.  

Ok! Let's code it. 

# Step
There are 3 steps in this method.  

In the first step, we need to randomly select a word and set a threshold about the similarity to prevent the synonym from mismatching.  

In the second step, according to the word, we can find out the top 10 similar synonyms and use the threshold to remove the similar synonyms below the threshold.  

In the third step, randomly select the synonym from the previous result to replace the source word.  

# Requirement
Please install packages by the following list.
```python
pip install --upgrade gensim numpy
```

# Code

## import
```python
#import
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import numpy as np
import random
import string
```

## class
```python
# class


class RandomSynonymReplacement:
    def __init__(self, corpus: str, similarity_threshold: float) -> None:
        self.model = Word2Vec(api.load(corpus))  # create the model of Word2Vec
        self.similarity_threshold = similarity_threshold    # set the threshold

    def __call__(self, text: str) -> str:
        # Split the input text with spaces to get each word
        # and check if the last character is a punctuation mark
        if text[-1] in string.punctuation:
            words = text[:-1].split(' ')
        else:
            words = text.split(' ')

        # randomly select a word and replace it with a synonym
        for word_index in random.sample(range(len(words)), len(words)):
            word = words[word_index]
            # turn the selected word to lower case
            # and check it whether exist in the vocabulary of the Word2Vec model
            if word.lower() in self.model.wv.key_to_index:
                # get similarity word by the model of Word2Vec
                # and put it to numpy array
                similarity_word = np.array(
                    self.model.wv.most_similar(word.lower()))
                # get the similarity from similarity_word
                similarity = similarity_word[:, 1].astype(np.float)
                # get the index with similarity above the threshold
                similarity_index = np.where(
                    similarity >= self.similarity_threshold)[0]
                # check the length of similarity_index
                if len(similarity_index):
                    # randomly select the synonym
                    words[words.index(word)] = random.sample(
                        list(similarity_word[similarity_index, 0]), 1)[0]
                    # check if the last character is a punctuation mark
                    if text[-1] in string.punctuation:
                        return ' '.join(words)+text[-1]
                    else:
                        return ' '.join(words)
        return text
```

## call
```python
if __name__ == '__main__':
    # create a class of RandomSynonymReplacement
    random_synonym_replacement = RandomSynonymReplacement(
        corpus='text8', similarity_threshold=0.5)

    # define a string
    text = 'Hello, World!'

    # check the result
    print(text)
    print(random_synonym_replacement(text=text))
```

## result
```python
Hello, World!
Hello, europe!
```

## full version
The full version of code is here: [https://github.com/fastyangmh/toolkit/blob/main/Python/RandomSynonymReplacement.py](https://github.com/fastyangmh/toolkit/blob/main/Python/RandomSynonymReplacement.py)

# Conclusion
If you have any questions, please feel free to contact me by email.

# Reference
[What is Gensim?](https://radimrehurek.com/gensim/intro.html)  
[NumPy](https://numpy.org/)  
[Data Augmentation in Natural Language Processing](https://maelfabien.github.io/machinelearning/NLP_8/#when-should-we-use-data-augmentation)  
[NLP Data Augmentation 常見方法](https://marssu.coderbridge.io/2020/10/26/nlp-data-augmenatation-%E5%B8%B8%E8%A6%8B%E6%96%B9%E6%B3%95/)  