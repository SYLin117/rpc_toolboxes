##################################################################################
import os
from multiprocessing import Pool, Value  # , set_start_method
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def func(x):
    for i in range(x):
        assert i == i
        with cnt.get_lock():
            cnt.value += 1
            print(f'{os.getpid()} | counter incremented to: {cnt.value}\n')


def init_globals(counter):
    global cnt
    cnt = counter


if __name__ == '__main__':

    # set_start_method('spawn')

    cnt = Value('i', 0)
    iterable = [10000 for _ in range(10)]

    # with Pool(processes=3, initializer=init_globals, initargs=(cnt,)) as pool:
    #     pool.map(func, iterable)
    jobs = []
    with ProcessPoolExecutor(max_workers=4, initializer=init_globals, initargs=(cnt,)) as executor:
        for _ in range(10):
            job = executor.submit(func, (10000))
            jobs.append(job)
        for job in as_completed(jobs):
            # result = job.result()
            jobs.remove(job)

    assert cnt.value == 100000
##################################################################################

# from multiprocessing import Process, Value, Array
#
# def f(n, a):
#     n.value = 3.1415927
#     for i in range(len(a)):
#         a[i] = -a[i]
#
# if __name__ == '__main__':
#     num = Value('d', 0.0)
#     arr = Array('i', range(10))
#
#     p = Process(target=f, args=(num, arr))
#     p.start()
#     p.join()
#
#     print(num.value)
#     print(arr[:])
