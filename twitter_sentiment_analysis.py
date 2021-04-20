import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string


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


if __name__ == '__main__':
    # Download dataset
    nltk.download('stopwords')
    nltk.download('twitter_samples')
    all_positive_samples = twitter_samples.strings('positive_tweets.json')
    all_negative_samples = twitter_samples.strings('negative_tweets.json')
    print('Positive {}: Negative {}'.format(
        len(all_positive_samples), len(all_negative_samples)))
