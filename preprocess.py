# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def setup_nltk():
    print("Загрузка NLTK-ресурсов (stopwords, wordnet)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK-ресурсы загружены.")


def clean_tweet(text):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'\@\w+', '', text)

    text = text.replace('#', '')

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in cleaned_tokens]

    return " ".join(lemmatized_tokens)


if __name__ == '__main__':

    setup_nltk()
    test_text = "Oh @Jack, I just LOVE this new #Twitter feature! It's SO cool. Check it out: https://example.com"
    print(f"Оригинал: {test_text}")
    print(f"Очищенный: {clean_tweet(test_text)}")