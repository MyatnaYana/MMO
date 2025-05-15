import spacy
from transformers import pipeline
from spacy import displacy
import warnings

# Игнорировать предупреждения о deprecated np.bool8
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Загрузка языковой модели для русского языка (spaCy)
try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    print("Ошибка: Модель 'ru_core_news_sm' не найдена. Установите её с помощью: python -m spacy download ru_core_news_sm")
    exit(1)

# Загрузка моделей для анализа тональности и эмоций
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
except Exception as e:
    print(f"Ошибка при загрузке модели для анализа тональности: {e}")
    exit(1)

try:
    emotion_analyzer = pipeline("text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection", return_all_scores=True)
except Exception as e:
    print(f"Ошибка при загрузке модели для анализа эмоций: {e}")
    exit(1)

# Входные тексты (отзывы)
texts = [
    "Этот продукт просто потрясающий, я так рад, что купил его!",
    "Ужасное качество, быстро сломался, очень разочарован.",
    "Неплохой товар, но доставка была медленной."
]

# Обработка каждого отзыва
for idx, text in enumerate(texts, 1):
    print(f"\n=== Обработка отзыва {idx}: '{text}' ===")
    
    # 1. Токенизация
    doc = nlp(text)
    print("\nТокенизация:")
    tokens = [token.text for token in doc]
    print("Токены:", tokens)
    
    # 2. Частеречная разметка
    print("\nЧастеречная разметка:")
    pos_tags = [(token.text, token.pos_, spacy.explain(token.pos_)) for token in doc]
    for token, pos, explanation in pos_tags:
        print(f"Слово: {token}, Часть речи: {pos} ({explanation})")
    
    # 3. Лемматизация
    print("\nЛемматизация:")
    lemmas = [(token.text, token.lemma_) for token in doc]
    for token, lemma in lemmas:
        print(f"Слово: {token}, Лемма: {lemma}")
    
    # 4. Анализ тональности
    print("\nАнализ тональности:")
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_label = sentiment_result['label'].upper()  # Приводим к верхнему регистру для консистентности
    sentiment_score = sentiment_result['score']
    print(f"Тональность: {sentiment_label}, Уверенность: {sentiment_score:.4f}")
    
    # 5. Извлечение эмоций
    print("\nИзвлечение эмоций:")
    emotion_results = emotion_analyzer(text)[0]
    for emotion in emotion_results:
        print(f"Эмоция: {emotion['label']}, Вероятность: {emotion['score']:.4f}")
    
    # 6. Синтаксический разбор
    print("\nСинтаксический разбор:")
    dependencies = [(token.text, token.dep_, spacy.explain(token.dep_), token.head.text) for token in doc]
    for token, dep, explanation, head in dependencies:
        print(f"Слово: {token}, Зависимость: {dep} ({explanation}), Главное слово: {head}")

    # Визуализация синтаксического разбора
    print("\nВизуализация синтаксического разбора сохранена в HTML.")
    html = displacy.render(doc, style="dep", jupyter=False, options={"compact": True, "distance": 90})
    with open(f"dependency_parse_review_{idx}.html", "w", encoding="utf-8") as f:
        f.write(html)

print("\nОбработка завершена. HTML-файлы с визуализацией сохранены.")