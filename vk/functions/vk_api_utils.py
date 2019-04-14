import urllib.request, json
import numpy as np
import time

client_id = '6610408' # ID приложения
client_secret = 'IXIehCixfDd3rHiXxOHn' # Защищённый ключ
service_key = '6523667d6523667d6523667daa6547bb95665236523667d3e0a32345be0383eea29fc0a' # Сервисный ключ доступа
version = '5.92' # Версия API

user_token = '22e85cc11336a5bb507f627f314913f440a3bdede618bc1aaa649c903a3b648c91c8039bcd449a1f4aed0' # Пользовательский ключ доступа

sleep_time = 10 # 5 min

def SendRequest(url):
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
        return data
    
def MethodRequest(method_name, params, access_token, version):
    params_string = ''
    for key, value in params.items():
        params_string += key + '=' + value + '&'
    request = 'https://api.vk.com/method/' + method_name + '?' + params_string[:-1] + '&access_token=' + access_token + '&v=' + version
    return SendRequest(request)

def UserTokenURL():
    authcode_flow_user = 'https://oauth.vk.com/authorize?client_id=' + client_id +\
    '&redirect_uri=https://oauth.vk.com/blank.html&scope=wall&response_type=token&v=' + version
    
    with urllib.request.urlopen(authcode_flow_user) as url:
        print(url.url)

def SaveJSON(data, filename):
    with open(filename + '.json', 'w') as outfile:
        json.dump(data, outfile)