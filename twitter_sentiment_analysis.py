import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':
    # Download dataset
    nltk.download('stopwords')
    nltk.download('twitter_samples')
    all_positive_samples = twitter_samples.strings('positive_tweets.json')
    all_negative_samples = twitter_samples.strings('negative_tweets.json')
    print('Positive {}: Negative {}'.format(
        len(all_positive_samples), len(all_negative_samples)))
