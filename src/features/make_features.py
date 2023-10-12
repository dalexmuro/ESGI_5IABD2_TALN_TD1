import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('french'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = ' '.join([w.lower() for w in word_tokens if not w.lower() in stop_words])
    return filtered_sentence


def stemming(sentence):
    word_tokens = word_tokenize(sentence)
    stemmer = SnowballStemmer(language='french')
    return ' '.join([stemmer.stem(X) for X in word_tokens])


def remove_punct(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = ' '.join([w for w in word_tokens if not w in string.punctuation])
    return filtered_sentence


def remove_numbers(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = ' '.join([w for w in word_tokens if not w.isnumeric()])
    return filtered_sentence


def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"]
    X = [remove_stopwords(i) for i in X]
    X = [stemming(i) for i in X]
    X = [remove_punct(i) for i in X]
    X = [remove_numbers(i) for i in X]


    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
