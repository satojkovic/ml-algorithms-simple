import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import os


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s+]', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags, only removing hash #
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweet_clean.append(stem_word)
    return tweet_clean


def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for tweet, y in zip(tweets, yslist):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def load_words(words_txt, encoding='utf-8'):
    words = set()
    with open(words_txt, 'r', encoding=encoding) as f:
        for line in f:
            if re.match(r'^;', line) or line == '\n':
                continue
            words.add(line.strip())
    return words


if __name__ == '__main__':
    # Download dataset
    nltk.download('stopwords')
    nltk.download('twitter_samples')
    all_positive_samples = twitter_samples.strings('positive_tweets.json')
    all_negative_samples = twitter_samples.strings('negative_tweets.json')
    print('Positive {}: Negative {}'.format(
        len(all_positive_samples), len(all_negative_samples)))
