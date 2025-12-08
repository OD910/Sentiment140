import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def setup_nltk():
    """
    Скачивает необходимые базы данных NLTK, если их нет локально.
    stopwords - список слов типа 'the', 'is', 'in'.
    wordnet - лексическая база данных для лемматизации.
    """
    print("Загрузка NLTK-ресурсов (stopwords, wordnet)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK-ресурсы загружены.")


def clean_tweet(text):
    """
    Очищает сырой текст твита.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем ссылки (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Удаляем упоминания пользователей (@username)
    text = re.sub(r'\@\w+', '', text)

    # Удаляем знак решетки, но оставляем само слово тега
    text = text.replace('#', '')

    # Удаляем все символы, кроме английских букв и пробелов.
    # Это удаляет цифры, смайлики (если они не обработаны заранее) и пунктуацию.
    text = re.sub(r'[^a-z\s]', '', text)

    # Разбиваем на токены (слова)
    tokens = text.split()

    # Фильтруем стоп-слова (удаляем 'the', 'is', 'a' и т.д., так как они не несут эмоциональной окраски)
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    # Лемматизация: приводим слова к нормальной форме (running -> run, better -> good)
    # Это помогает уменьшить размер словаря.
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in cleaned_tokens]

    return " ".join(lemmatized_tokens)


if __name__ == '__main__':
    # Блок для тестирования файла отдельно
    setup_nltk()
    test_text = "Oh @Jack, I just LOVE this new #Twitter feature! It's SO cool. Check it out: https://example.com"
    print(f"Оригинал: {test_text}")
    print(f"Очищенный: {clean_tweet(test_text)}")
    # Ожидаемый результат: "oh love new twitter feature cool check"