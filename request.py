import requests
import json
import re

def validate_response(response):
    """
    Проверяет, соответствует ли JSON-ответ требованиям:
    1. Это должен быть массив JSON объектов.
    2. Каждый объект должен содержать поля 'number' (целое число) и 'name' (строка).
    """
    try:
        data = json.loads(response)['choices'][0]['message']['content']
        data = json.loads(data)
        # Проверка, что это список
        if isinstance(data, dict) and len(data) == 2:
            data = [data]
        if not isinstance(data, list):
            print("First False")
            return False
        
        pattern = r'^\d+_\d+_\d+$' # \d означает любую цифру, + означает одно или более повторений.
    
        for item in data:
            if not isinstance(item, dict):
                print("Second False")
                return False
            if "number" not in item or "name" not in item:
                print("Third False")
                return False
            if not bool(re.match(pattern, item['number'])) or not isinstance(item["name"], str):
                print("Fourth False")
                return False
            print(type(item['name']), item['name'])
        
        return True
    except json.JSONDecodeError:
        print('false')
        print(data)
        return False

def request_cluster_names(api, prompt, claster_dict):

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps({
    "model": "GigaChat",
    "messages": [
        {
        "role": "system",
        "content": prompt
        },
        {
        "role": "user",
        "content": claster_dict
        }
    ],
    "stream": False,
    "update_interval": 0
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {api}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    return response.text

def generate_api_key():
    try:
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        payload='scope=GIGACHAT_API_PERS'
        headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '7cd0d62f-d2cc-4667-8e58-d43aad551644',
        'Authorization': 'Basic NWM5MjdiZTEtMzA2MS00NjdlLWE2NjAtYzk3NTQ1NDQ1OGVmOjNlYzIzNTU4LWU5NzQtNDFkOC1hNmRlLTQzMzYyNzVlNzVlZQ=='
        }

        response = requests.request("POST", url, headers=headers, data=payload, verify=False)   
        response = json.loads(response.text)
        
        if 'access_token' not in response.keys():
            raise Exception(json.dumps(response))
        
        return response['access_token'] 
    except Exception as e:
        print(f'Ошибка при генерации API ключа. {e}')

def name_summarize(nodes, prompt_file):
    api = generate_api_key()
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    id_list = list(nodes.keys())
    for i in range(len(nodes) // 5 + 1 - (len(nodes) % 5 == 0)):
        cur_prompt = ''
        for node_ind in id_list[5*i:5*i+5]:
            cur_prompt += '\nКластер' + ' ' + str(node_ind) + ':\n' + '\n'.join(nodes[node_ind]['data']) + '\n'

        while True:
            response = request_cluster_names(api, prompt, cur_prompt) 
            if validate_response(response):
                break

        parsed_response = json.loads(response)['choices'][0]['message']['content']
        parsed_response = json.loads(parsed_response)
        if isinstance(parsed_response, dict) and len(parsed_response) == 2:
            parsed_response = [parsed_response]
        
        for node in parsed_response:
            nodes[node['number']]['data'] = node['name']