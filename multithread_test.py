from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from random import randint


def say_hello_to(idx, name):
    val = randint(1, 10)
    time.sleep(val)
    return f'Hi, {idx, name}'


names = ['John', 'Ben', 'Bill', 'Alex', 'Jenny'] * 10

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for idx, n in enumerate(names):
        future = executor.submit(say_hello_to, idx, n)
        # print(type(future))
        futures.append(future)

    for future in as_completed(futures):
        print(future.result())

    print("all done.")
