import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.cluster import KMeans
import copy
import json
import numpy as np
import simple_parser as parse


# Загрузка необходимых ресурсов


    

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()


# Предварительная обработка текста
def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(text.lower())

    # Фильтрация токенов: оставляем только существительные и выполняем лемматизацию
    nouns = [
        morph.parse(word)[0].normal_form  # Лемматизация
        for word in tokens
        if word.isalnum() and word not in stop_words and morph.parse(word)[0].tag.POS == 'NOUN'
    ]

    return nouns


# Функция для выделения ключевых слов с помощью LDA
def extract_keywords(text, n_keywords):
    # Подготовка данных для Word2Vec
    processed_paragraph = preprocess_text(text)

    # Подготовка данных для LDA
    dictionary = Dictionary([processed_paragraph])
    corpus = [dictionary.doc2bow(processed_paragraph)]

    # Обучение модели LDA
    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10, random_state=42)

    # Получение ключевых слов
    keywords = lda_model.get_topic_terms(0, n_keywords)  # Получаем n_keywords ключевых слов
    keywords_list = [dictionary[id] for id, _ in keywords]

    return ' '.join(keywords_list)


def get_paragraphs_id(paragraphs_keywords):
    result = []
    for id_and_keywords in paragraphs_keywords:
        result.append(list(id_and_keywords.keys())[0])
    return result


def make_paragraphs_clustering(paragraphs_keywords, num_clusters=3):
    model_w2v = Word2Vec(paragraphs_keywords, vector_size=100, window=3, min_count=1, workers=4)

    paragraphs_vectors = []
    for p in paragraphs_keywords:
        vectors = [model_w2v.wv[word] for word in p if word in model_w2v.wv]
        if vectors:  # Проверяем, что список не пуст
            vector = np.mean(vectors, axis=0)
            paragraphs_vectors.append(vector)
        else:
            # Можно добавить нулевой вектор или игнорировать
            paragraphs_vectors.append(np.zeros(model_w2v.vector_size))  # Заменяем на нулевой вектор

    if len(paragraphs_vectors) > 0:  # Проверяем, есть ли векторы для кластеризации
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(paragraphs_vectors)
        labels = kmeans.labels_


    paragraphs_clusters = list(zip(labels, get_paragraphs_id(paragraphs_keywords)))

    label_with_ids = {}
    for label, id in paragraphs_clusters:
        if label not in label_with_ids:
            label_with_ids[label] = []
        label_with_ids[label].append(id)

    return label_with_ids


def get_clustered_keywords(text_with_id, clustered_paragraphs):
    clustered_keywords = dict()

    for cluster in clustered_paragraphs:
        join_in_cluster = None
        for text_id in clustered_paragraphs[cluster]:
            if join_in_cluster:
                ' '.join([join_in_cluster, text_with_id[text_id]])
            else:
                join_in_cluster = text_with_id[text_id]

        keywords = extract_keywords(join_in_cluster, 5)
        clustered_keywords[cluster] = keywords

    return clustered_keywords


def remove_clustering_words(text_with_id, clustered_paragraphs, clustered_keywords):
    for cluster in clustered_paragraphs:
        for text_id in clustered_paragraphs[cluster]:
            for word_to_remove in clustered_keywords[cluster].split():
                text_with_id[text_id] = text_with_id[text_id].replace(word_to_remove, "")









def convert_to_standard_types(data):
    """Преобразует данные в стандартные типы Python для сериализации в JSON."""
    if isinstance(data, dict):
        return {key: convert_to_standard_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_standard_types(item) for item in data]
    elif isinstance(data, np.integer):  # Проверяем на тип int32
        return int(data)  # Преобразуем в обычный int
    elif isinstance(data, np.floating):  # Проверяем на тип float
        return float(data)  # Преобразуем в обычный float
    else:
        return data  # Возвращаем без изменений, если тип стандартный


def process_text(text_with_id, num_iterations, current_iteration, answer, n_keywords=5, num_clusters=3, prev_num=None):
    paragraphs_keywords = []
    for text_id, content in text_with_id.items():
        if len(content):
            keywords = extract_keywords(content, n_keywords)
            paragraphs_keywords.append({text_id: keywords})
    

    
    if num_iterations <= current_iteration or (num_clusters > len(paragraphs_keywords) and prev_num != None):
        return

    if prev_num != None:
        clustered_paragraphs = make_paragraphs_clustering(paragraphs_keywords, num_clusters)
    else:
        clustered_paragraphs = make_paragraphs_clustering(paragraphs_keywords, 1)
    
    clustered_keywords = get_clustered_keywords(text_with_id, clustered_paragraphs)
    for cluster, keywords in clustered_keywords.items():
        small_answer = dict()
        small_answer["num"] = cluster
        small_answer["data"] = keywords
        small_answer["level"] = current_iteration
        small_answer["prev_num"] = prev_num
        answer.append(small_answer)


    
    remove_clustering_words(text_with_id, clustered_paragraphs, clustered_keywords)
    for cluster, paragraph_ids in clustered_paragraphs.items():
        cluster_text_with_id = dict()
        for paragraph_id in paragraph_ids:
            cluster_text_with_id[paragraph_id] = text_with_id[paragraph_id]
        process_text(cluster_text_with_id, num_iterations, current_iteration + 1, answer, n_keywords, num_clusters=num_clusters, prev_num=cluster)



def generate_json_output(names):
    text_with_id = {
        0 : "Современная наука стремительно развивается, открывая новые горизонты в понимании мира. Одним из наиболее впечатляющих достижений является \
        создание вакцин против COVID-19 за рекордно короткие сроки, что продемонстрировало возможности мРНК-технологий. Ученые продолжают исследовать \
        генетический материал и его влияние на здоровье человека, что открывает новые пути для лечения наследственных заболеваний. В дополнение к этому, \
        исследования в области квантовых технологий обещают революционизировать вычисления и коммуникации, что имеет потенциал изменить множество отраслей.",

        1 : "В последние годы политическая ситуация в мире претерпевает значительные изменения. Геополитическая напряженность, связанная с конфликтами и \
        экономическими санкциями, формирует новые альянсы и перераспределяет глобальные силы. Важными вопросами становятся вопросы безопасности, миграции\
        и изменения климата, которые требуют координации действий на международном уровне. Политики разных стран все чаще обращаются к национальным интересам, \
        что приводит к внутренним протестам и дискуссиям о будущем демократии в разных уголках мира.",

        2 : "Технологический прогресс изменяет нашу жизнь на всех уровнях. Искусственный интеллект и машинное обучение находят все более широкое применение в бизнесе, \
        здравоохранении и образовании. Системы, способные обрабатывать огромные объемы данных, помогают принимать более обоснованные решения и прогнозировать \
        будущие тренды. Кроме того, развитие интернета вещей (IoT) способствует созданию 'умных' домов и городов, повышая уровень комфорта и безопасности для жителей.\
        В то же время, вопросы этики и безопасности данных становятся все более актуальными в условиях цифровизации общества.",

        3 : "Глава Meta (компания-владелец соцсетей Facebook и Instagram, признана в России экстремистской и запрещена) Марк Цукерберг\
        занял второе место в списке самых богатых людей мира по версии Bloomberg, обойдя основателя Amazon Джеффа Безоса на $1 млрд. \
        Состояние Цукерберга вечером 3 октября достигло $206 млрд (по версии агентства, отметку в $200 млрд оно впервые преодолело \
        в конце сентября). Это произошло на фоне роста акций Meta, вечером в четверг они достигли рекордных $582,77. \
        Акции компании растут с начала года. Meta тратит значительные средства на центры обработки данных и вычислительные мощности, \
        компания также значительно продвинулась в работе над рядом значимых проектов, отмечает Bloomberg. Например, в сентябре она представила \
        очки дополненной реальности Orion.",

        4 : "Современные технологии значительно изменили повседневную жизнь людей. Инновации, такие как смартфоны, интернет и социальные сети, \
        создали новые способы общения и взаимодействия. Люди могут оставаться на связи с друзьями и семьей независимо от расстояния, что улучшает \
        социальные связи. Однако с этим возникают и новые вызовы, такие как необходимость защиты личной информации и управление зависимостью от технологий.",

        5 : "Устойчивое развитие стало важным понятием в области экологии и экономики. Оно подразумевает использование ресурсов таким образом, чтобы удовлетворить \
        потребности текущего поколения, не ставя под угрозу возможности будущих поколений. Это включает в себя переход на возобновляемые источники энергии, сокращение \
        отходов и внедрение практик, способствующих сохранению экосистем. Устойчивое развитие требует совместных усилий со стороны правительств, бизнеса и общества.",

        6 : "Искусственный интеллект (ИИ) стремительно развивается и начинает занимать ключевые позиции в различных отраслях. Алгоритмы машинного обучения и глубокого \
        обучения позволяют компьютерам анализировать большие объемы данных, что приводит к более эффективным решениям в медицине, финансах, производстве и других областях. \
        Однако с ростом ИИ также возникают этические вопросы, касающиеся его использования, влияния на рабочие места и ответственности за принимаемые решения.",

        7 : "Психическое здоровье становится все более важной темой в современном обществе. С ростом осведомленности о психических расстройствах, таких как депрессия и \
        тревожные расстройства, увеличивается число людей, обращающихся за помощью. Это приводит к необходимости развития более доступных и качественных \
        психотерапевтических услуг. Важно также развивать культуры поддержки и понимания, чтобы люди не боялись открыто говорить о своих проблемах.",

        8 : "Искусство и культура играют ключевую роль в жизни общества, влияя на наше восприятие мира и друг друга. Они помогают выражать идеи, эмоции и идентичность, формируя \
        культурные традиции и ценности. Современные формы искусства, такие как цифровое искусство и уличное искусство, открывают новые возможности для самовыражения и диалога \
        между различными сообществами. Поддержка культурных инициатив и искусства способствует развитию креативности и социальной сплоченности."
    }


    answer = []
    # уровень вложенности
    num_iter = 3
    n_keywords = 5
    num_clusters = 3
    if names:
        a = parse.File_parser(path='./cache/')
        a.new_files_parse(names)
        text_with_id = a.parsed_res[0]
    
    process_text(copy.deepcopy(text_with_id), num_iter, 0, answer, n_keywords, num_clusters)

    # Генерация JSON-файла
    converted_answer = convert_to_standard_types(answer)  # Преобразуем answer перед записью
    
    with open('output.json', 'w', encoding='utf-8') as json_file:
        json.dump(converted_answer, json_file, ensure_ascii=False, indent=4)

