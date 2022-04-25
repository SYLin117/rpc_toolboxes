import requests
import json
from tqdm import tqdm


def delete_all_cat():
    login_request = requests.post('http://192.168.50.75:5000/api/user/login',
                                  json={'username': 'admin', 'password': 'password'})

    if login_request.status_code == 200:
        for id in range(202):
            request = requests.delete(f'http://192.168.50.75:5000/api/category/{id}',
                                      cookies=login_request.cookies)
            if request.status_code != 200:
                print(request.text)


def add_cat_all():
    with open('class_initializer.json', 'r', encoding='UTF-8') as fid:
        json_data = json.load(fid)
    categories = json_data['categories']
    # print(len(categories))
    login_request = requests.post('http://192.168.50.75:5000/api/user/login',
                                  json={'username': 'admin', 'password': 'password'})
    if login_request.status_code == 200:
        id = 1
        for cat in tqdm(categories):
            add_cat_request = requests.post('http://192.168.50.75:5000/api/category',
                                            json={'id': id, 'name': cat['name'],
                                                  'supercategory': cat['supercategory'], },
                                            cookies=login_request.cookies)
            id += 1
            if add_cat_request.status_code != 200:
                print(add_cat_request.text)


if __name__ == "__main__":
    add_cat_all()
    # delete_all_cat()
