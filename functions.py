import pandas as pd
from razdel import tokenize as razdel_tokenize
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import numpy as np
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
import pickle
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab


def text_lower(text):
    return text.lower()

def clean_stopwords(text):
    stopwordlist = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
                    'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
                    'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне',
                    'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему',
                    'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
                    'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
                    'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего',
                    'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней',
                    'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
                    'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
                    'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого','это',
                    'эта','мы','они','для','с','наш','кто','от','тех','нам','от','наша',
                    'нашей','наша','своя','свои','своих','должны','должен','стоит','нашими',
                    '-нашей','эти','этой','нас']
    STOPWORDS = set(stopwordlist)
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Убираем пунктуацию 
def clean_puctuations(text):
    english_puctuations = string.punctuation
    translator = str.maketrans('','', english_puctuations)
    return text.translate(translator)

# Повторяющиеся знаки
def clean_repeating_characters(text):
    return re.sub(r'(.)1+', r'1', text)

# Ссылки тоже чистим
def clean_URLs(text):
    return re.sub(r"((www.[^s]+)|(http\S+))","",text)

# Удаляем все цифры
def clean_numeric(text):
    return re.sub('[0-9]+', '', text)

# Разбиваем текст
def tokenize_tweet(text):
    return [token.text for token in razdel_tokenize(text)]

# Убираем окончания  
def text_stemming(words):
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(word) for word in words]

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
# превращаем в неизмененные слова
def text_lemmatization(text):
    doc = Doc(text)
    doc.segment(segmenter)  # вот это важно!
    doc.tag_morph(morph_tagger)
    
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)
    return lemmas

def preprocess(text_series):
    text_series = text_series.apply(clean_URLs)
    text_series = text_series.apply(clean_numeric)
    text_series = text_series.apply(clean_puctuations)
    text_series = text_series.apply(text_lower)
    text_series = text_series.apply(clean_stopwords)
    text_series = text_series.apply(tokenize_tweet)
    text_series = text_series.apply(text_stemming)
    text_series = text_series.apply(lambda x: " ".join(x))
    text_series = text_series.apply(text_lemmatization)
    text_series = text_series.apply(lambda x: " ".join(x))     
    return text_series

# Ввод для текста
def custom_input_prediction(text):
    import nltk
    nltk.download('omw-1.4')

    try:
        # Преобразование текста в серию
        text_series = pd.Series(text)
        text_series = preprocess(text_series)

        # Проверка на пустой или недопустимый текст
        if text_series[0] is None or text_series[0].strip() == "":
            raise ValueError("Преобразованный текст пуст или недопустим")

        text = [text_series[0]]

        # Загрузка векторизатора
        with open(r"C:\Users\marik\OneDrive\Desktop\AE\tdf_vectorizer", "rb") as vec_file:
            vectoriser = pickle.load(vec_file)

        # Преобразование текста в вектор
        text_vector = vectoriser.transform(text)

        # Загрузка модели
        with open(r"C:\Users\marik\OneDrive\Desktop\AE\model.bin", "rb") as model_file:
            model = pickle.load(model_file)

        # Предсказание
        prediction = model.predict(text_vector)[0]

        # Интерпретации предсказания
        interpretations = {
            0: "ethnicity",
            1: "extremist",
            2: "non_extremist",
            3: "political",
            4: "religious"
        }

        # Возвращаем интерпретацию
        result = interpretations.get(prediction, "Unknown")
        print(f"Предсказание: {result}")  # Добавим вывод для отладки
        return result

    except Exception as e:
        # Логирование ошибок
        print(f"Произошла ошибка при анализе текста: {e}")
        return None


