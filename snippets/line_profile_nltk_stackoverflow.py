import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from tqdm import tqdm


@profile
def profile_nltk():
    df = pd.read_csv('IMDB_Dataset.csv')  # (50000, 2)
    filtered = []
    reviews = df['review'][:4000]
    corpus = []
    for i in tqdm(range(len(reviews))):
        rev = re.sub('[^a-zA-Z]', ' ', df['review'][i])
        rev = rev.lower()
        rev = rev.split()
        filtered = []
        for word in rev:
            if word not in stopwords.words("english"):
                word = PorterStemmer().stem(word)
                filtered.append(word)
        filtered = " ".join(filtered)
        corpus.append(filtered)


if __name__ == '__main__':
    profile_nltk()
