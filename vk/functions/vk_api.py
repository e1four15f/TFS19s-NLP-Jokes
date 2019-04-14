import numpy as np
import time

from functions.vk_api_utils import *


def GroupsSearch(text, count=1000): 
    method_name = 'groups.search'
    params = {
        'q': urllib.parse.quote(text), 
        'type': 'groups',  #Возможные значения: group, page, event.
        'count': str(count), #max 1000
    }
    data = MethodRequest(method_name, params, user_token, version)
    
    try:
        print(data['error']['error_code'], data['error']['error_msg'])
        if data['error']['error_code'] == 6:
            time.sleep(np.random.randint(5, sleep_time))
            return GroupsSearch(text, count)
        return []
    except KeyError:
        pass
    
    #Обработка json
    group_ids = np.array([i['screen_name'] for i in data['response']['items']])
    return group_ids


def WallGet(domain, count=100, offset=0): 
    method_name = 'wall.get'
    params = {
        'domain': domain, 
        'count': str(count),
        'offset': str(offset),
    }
    data = MethodRequest(method_name, params, user_token, version)
    
    try:
        print(data['error']['error_code'], data['error']['error_msg'])
        return []
    except KeyError:
        pass
    
    return data['response']['items']