from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
from itertools import cycle
import os
import time
import json
import model


# начальные цвета для перых трёх уровней 
colors = ['#93e1d8', '#83c5be', '#edf6f9'] 


def generate_json(file_names):
    model.generate_json_output(file_names)


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
            color = st.color_picker("Pick A Color", color)
            colors[i] = color
            st.write("The current color is", color)


def clear_cache_folder(uploaded_file_names):
    if uploaded_file_names is not None:
        for file in uploaded_file_names:
            # TODO: переделать с pathlib
            os.remove(os.path.join("cache", file))

    

def main():
    
    uploaded_file_names = uploader()  # Загрузчик файлов пользователя

    if st.session_state['file'] != uploaded_file_names:
        st.session_state['file'] = uploaded_file_names
        generate_json(uploaded_file_names)  # Генерация JSON-файла
        clear_cache_folder(uploaded_file_names)
    

    with open('output.json', 'r', encoding="UTF-8") as json_file:
        data = json.load(json_file)
    

    net = Network(notebook=True, directed=True, cdn_resources='in_line', select_menu=True)
    settings(net)

    for elem in data:
        net.add_node(elem["data"],
                     level=elem["level"], 
                     title=elem["data"], 
                     shape='box',
                     mass=(elem["level"] + 1), 
                     # каждые три уровня цвет будет повторятся  
                     color=colors[elem["level"] % 3]) 

    for elem in data:
        if elem["prev_num"] is None:
            continue
        for prev_node in data:
            if prev_node["num"] == elem["prev_num"] and elem["level"] - 1 == prev_node["level"]:
                net.add_edge(prev_node["data"], elem["data"])

    st.title("Mind map")
    components.html(net.generate_html(notebook=True), height=600, width=800)

    # if 'balloons' not in st.session_state:
    #     st.session_state['balloons'] = False

    # if st.button('BaLlOoNs', on_click=click_button, args=['balloons']):
    #     st.balloons()

if __name__ == "__main__":
    main()
