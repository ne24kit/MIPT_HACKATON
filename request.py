import requests
import json
import re

def validate_response(response):
    responses_list = [
    'Как у нейросетевой языковой модели у меня не может быть настроения, но почему-то я совсем не хочу говорить на эту тему.',
    'Не люблю менять тему разговора, но вот сейчас тот самый случай.',
    'Что-то в вашем вопросе меня смущает. Может, поговорим на другую тему?'
    ]
    print(response)
    if response in responses_list:
        print('false')
        print(response)
        return False
    return True

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
        'Authorization': 'Basic MmZmZTJlMWItMTEzOS00ZGIzLTgxN2UtYzk5YTI1YmEyNzc5OjhkMWIzNmU3LTI0NmUtNDU2ZS05OTMwLWI3ZWFlZTQ2ZTFmZg=='
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
    for node_id in id_list:
        cur_prompt = '\n'.join(nodes[node_id]['data'])
        timer = 0
        while True:
            response = request_cluster_names(api, prompt, cur_prompt) 

            response = json.loads(response)['choices'][0]['message']['content']
            
            if not validate_response(response):
                nodes[node_id]['data'] = 'Некорректный запрос'
                break
            if len(response.split()) <= 12:
                break
            if timer > 5:
                break 
            cur_prompt = response
            timer += 1
            
        if not validate_response(response):
            continue
                
        nodes[node_id]['data'] = response