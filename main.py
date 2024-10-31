from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import os
import time
import json
from model import generate_json


# начальные цвета для перых трёх уровней 
colors = ['#c6b6d6', '#efc4d6', '#ffddd6', '#fff6e6'] 


def generate_json_output():
    generate_json()


def disappearing_message(mess, sec=2):
    with st.spinner(mess):
        time.sleep(sec)

def uploader():
    uploaded_files = st.file_uploader(label="Upload .md files, limited to 200MB",
                                       type=['md'],
                                       accept_multiple_files=True,
                                       help="Choose and upload .md files")
    
    uploaded_file_names = []
    if uploaded_files is not None:
        for file in uploaded_files:
            if file.name not in uploaded_file_names:
                uploaded_file_names.append(file.name)
                # TODO: переделать с pathlib
                with open(os.path.join("cache", file.name), "wb") as f: 
                    f.write(file.getbuffer())

    if 'file' not in st.session_state:
        st.session_state['file'] = []

    return uploaded_file_names

def settings(net):
    with st.sidebar:
        st.title('Settings')
        net.toggle_physics(st.toggle("Physics off/on"))
        for i, color in enumerate(colors):       
            color = st.color_picker(f"pick a color for {i} level ", color)
            colors[i] = color
            st.write(color)


def clear_cache_folder(uploaded_file_names):
    if uploaded_file_names is not None:
        for file in uploaded_file_names:
            # TODO: переделать с pathlib
            os.remove(os.path.join("cache", file))

    

def main():
    
    uploaded_file_names = uploader()  # Загрузчик файлов пользователя
    if st.session_state['file'] != uploaded_file_names:
        print(f'Загружены файлы: {uploaded_file_names}')
        st.session_state['file'] = uploaded_file_names
        with st.spinner("Обработка файлов и генерация карты знаний..."):
            generate_json_output()  # Генерация JSON-файла
        clear_cache_folder(uploaded_file_names)
    

    with open('output_evg.json', 'r', encoding="UTF-8") as json_file:
        data = json.load(json_file)
        data = sorted(data, key=lambda x: (x['level'], x['prev_num'], x['num']))

    net = Network(notebook=True, directed=True, cdn_resources='in_line', select_menu=True)
    settings(net)

    # Создаём узлы и словарь для быстрого доступа по (level, num)
    node_dict = {}
    for id, elem in enumerate(data):
        net.add_node(n_id=id,
                    label=elem["data"],
                    level=elem["level"],
                    title=str(id),
                    shape='box',
                    mass=(elem["level"] + 1),
                    color=colors[elem["level"] % 4])
        
        # Сохраняем узел с уникальным ID
        elem['id'] = id
        node_dict[(elem["level"], elem["num"])] = id

    # Добавляем рёбра с использованием словаря
    for elem in data:
        if elem["prev_num"] is not None:
            prev_id = node_dict.get((elem["level"] - 1, elem["prev_num"]))
            if prev_id is not None:
                net.add_edge(prev_id, elem["id"])

    
    st.title("Mind map")
    components.html(net.generate_html(notebook=True), height=600, width=800)

    st.subheader("Поиск по ID узла")
    node_id = st.text_input("Введите ID узла:", help='Наведитесь на узел и вы увидите ID')

    if node_id:
        try:
            node = net.get_node(int(node_id))
            st.write(f"Название узла:\n {node['label']}")
            st.write("Этот узел содержит информацию в следующих файлах (в соответсвующих параграфах)")
            st.write(data[int(node_id)]['info'])
        except KeyError:
            st.write("Пожалуйста, введите существующий числовой ID")    
        except ValueError:
            st.write("Пожалуйста, введите допустимый числовой ID")


    
    # if 'balloons' not in st.session_state:
    #     st.session_state['balloons'] = False
    # if st.button('BaLlOoNs'):
    #     st.balloons()
    #     st.snow()

if __name__ == "__main__":
    main()
