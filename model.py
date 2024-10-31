import os
import re
import json
import numpy as np
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from request import name_summarize

morph = pymorphy3.MorphAnalyzer()

def process_markdown(text):
    text = re.sub(r'#\s*(.*)', r' заголовок: \1 ', text)  # Заголовки
    text = re.sub(r'\*\*(.*?)\*\*', r' выделенный: \1 ', text)  # Жирный текст
    text = re.sub(r'~~(.*?)~~', r' зачеркивание: \1 ', text)  # Зачеркнутый текст
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text)  # Убираем ненужные символы
    return text

def lemmatize(text):
    # Используем регулярное выражение, чтобы захватить слова и пробельные символы отдельно
    tokens = re.findall(r'\S+|\s+', text)
    lemmatized = [
        morph.parse(token)[0].normal_form if token.strip() else token  # Нормализуем только слова
        for token in tokens
    ]
    return ''.join(lemmatized)

def convert_to_standard_types(data):
    """Преобразует данные в стандартные типы Python для сериализации в JSON."""
    if isinstance(data, dict):
        return {key: convert_to_standard_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_standard_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data

def extract_headings_by_level(text, prev_level_info, level=1):    
    pattern = rf"(^{'#' * level} .+)(?=\n|$)\n((?:.*\n)*?(?=^#{{1,{level}}} |\Z))"
    matches = re.findall(pattern, text, re.MULTILINE)
    results = [(prev_level_info, match[0].strip(), lemmatize(match[1].strip())) for match in matches if match[1].strip()]
    return results

# Функция для кластеризации и рекурсивной обработки заголовков
def recursive_clustering(headings_content, nodes, level=1, prev_num=None):
    print(f"\nКластеризация уровня {level}")
    if not headings_content or len(headings_content) < 2:
        print(f"Недостаточно данных для кластеризации на уровне {level}. Пропуск.")
        return

    _, headings, contents = zip(*headings_content)
    if not any(contents):
        print(f"Уровень {level} содержит только пустые или стоп-слова. Пропуск.")
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents)

    # Определение оптимального количества кластеров
    silhouette_scores = []
    K = range(2, min(len(headings), 8))

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    # Проверка наличия оценок силуэта
    if not silhouette_scores:
        print(f"Недостаточно вариативности данных для кластеризации на уровне {level}. Пропуск.")
        return

    # Определение количества кластеров
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    print(f'Оптимальное количество кластеров для уровня {level}: {optimal_k}')

    # Кластеризация
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Рекурсивная обработка
    classified_headings = {i: [] for i in range(optimal_k)}
    for i, label in enumerate(labels):
        classified_headings[label].append(headings_content[i])

    # Вывод результатов и рекурсивный вызов для подзаголовков
    for cluster_id, entries in classified_headings.items():
        print(f'\nКластер {cluster_id + 1} уровня {level}:')
        
        additional_info = []
        headers = []
        sub_headings_content = []
        for prev_info, heading, content in entries:
            print(f' - Заголовок: {heading}')
            headers.append(heading)
            new_prev_info = prev_info + '\n' + heading
            additional_info.append(new_prev_info)
            sub_result = extract_headings_by_level(content, new_prev_info, level + 1)
            for res in sub_result:
                sub_headings_content.append(res)
        node = {}
        node["level"] = level
        node["num"] = cluster_id
        node["prev_num"] = prev_num
        node["data"] = headers
        node["info"] = additional_info
        nodes[f"{level}_{cluster_id}_{prev_num}"] = node
        # нужно запихнуть это в for, чтобы для каждого отельно sub_headings_content формировался и добавлять в некий результирующий список
        if sub_headings_content:
            recursive_clustering(sub_result, nodes, level + 1, prev_num=cluster_id)


def process_files(files, nodes, prev_num):
    # Чтение и обработка указанных файлов
    top_level_headings_content = []
    for file_path in files:
        with open(f"./cache/{file_path}", 'r', encoding='utf-8') as file:
            file_content = file.read()
            print(f"\nФайл: {file_path}")
            result = extract_headings_by_level(file_content, file_path, level=1)
            for res in result:
                top_level_headings_content.append(res)

    recursive_clustering(top_level_headings_content, nodes, level=1, prev_num=prev_num)


def generate_json():
    
    # Путь к директории с markdown файлами
    directory = './cache' 
    
    documents = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                file_names.append(filename)

    # Обработка документов
    processed_documents = [lemmatize(process_markdown(doc)) for doc in documents]

    # Применение TF-IDF для векторизации текстов
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_documents)
    
    nodes = {}

    # добавляем начальную ноду
    node = {}
    node["num"] = 0
    node["level"] = 0
    node["prev_num"] = None
    node["data"] = "Nothing yet"
    node["info"] = file_names
    nodes["0_0_0"] = node

    # Определение оптимального количества кластеров с помощью метода локтя
    silhouette_scores = []
    K = range(2, min(len(documents), 8))  

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
  
    # Выбор количества кластеров на основе максимального значения силуэта
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    print(f'Оптимальное количество кластеров: {optimal_k}')

    # Кластеризация с найденным оптимальным количеством кластеров
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(X)

    # Сохранение результатов кластеризации
    classified_files = {i: [] for i in range(optimal_k)}
    for i, label in enumerate(kmeans.labels_):
        classified_files[label].append(file_names[i])

    file_names = {}
    for key, files in classified_files.items():
        file_names[key] = []
        for file_name in files:
            file_names[key].append(file_name)

    # Вывод результатов без названий категорий
    for category, files in classified_files.items():
        node = {}
        print(f'Категория {category + 1}:')
        node["num"] = category
        node["level"] = 1
        node["prev_num"] = 0
        node["data"] = file_names[category]
        node["info"] = file_names[category]
        nodes[f"1_{category}_0"] = node
        process_files(files, nodes, category)

    name_summarize(nodes, './prompt.txt')

    converted_answer = convert_to_standard_types(nodes)
    with open('output.json', 'w', encoding='utf-8') as json_file:
        json.dump(list(converted_answer.values()), json_file, ensure_ascii=False, indent=4)