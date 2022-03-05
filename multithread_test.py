from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from random import randint
import multiprocessing

def say_hello_to(idx, name):
    val = randint(1, 10)
    time.sleep(val)
    return f'Hi, {idx, name}'


def dummy_def(id, value, lock):
    with lock:
        print(id)
        print(value)


x = [('synthesized_image_0', {195: 1, 159: 1, 72: 1, 65: 1, 125: 1, 120: 2}),
     ('synthesized_image_1', {77: 5, 160: 1, 80: 1, 91: 1, 84: 1, 120: 1, 110: 1, 105: 1}),
     ('synthesized_image_2', {13: 2, 61: 2, 127: 2, 185: 1, 104: 1, 195: 1, 79: 1, 122: 1}),
     ('synthesized_image_3', {12: 1, 30: 1, 102: 1, 44: 2, 91: 1}),
     ('synthesized_image_4', {56: 3, 142: 2, 95: 1, 5: 1, 73: 1, 103: 1, 149: 1, 152: 1}),
     ('synthesized_image_5', {116: 1, 158: 6, 193: 1, 168: 1, 88: 1}),
     ('synthesized_image_6', {55: 3, 112: 3, 20: 1, 32: 3}),
     ('synthesized_image_7', {87: 1, 100: 4, 103: 1, 155: 1, 94: 1, 135: 1, 99: 1, 17: 1, 93: 1}),
     ('synthesized_image_8', {59: 3, 33: 1, 190: 1, 56: 1, 4: 1, 132: 1}),
     ('synthesized_image_9', {183: 4, 79: 1, 113: 1, 148: 1}),
     ('synthesized_image_10', {195: 1, 159: 1, 72: 1, 65: 1, 125: 1, 120: 2}),
     ('synthesized_image_11', {77: 5, 160: 1, 80: 1, 91: 1, 84: 1, 120: 1, 110: 1, 105: 1}),
     ('synthesized_image_12', {13: 2, 61: 2, 127: 2, 185: 1, 104: 1, 195: 1, 79: 1, 122: 1}),
     ('synthesized_image_13', {12: 1, 30: 1, 102: 1, 44: 2, 91: 1}),
     ('synthesized_image_14', {56: 3, 142: 2, 95: 1, 5: 1, 73: 1, 103: 1, 149: 1, 152: 1}),
     ('synthesized_image_15', {116: 1, 158: 6, 193: 1, 168: 1, 88: 1}),
     ('synthesized_image_16', {55: 3, 112: 3, 20: 1, 32: 3}),
     ('synthesized_image_17', {87: 1, 100: 4, 103: 1, 155: 1, 94: 1, 135: 1, 99: 1, 17: 1, 93: 1}),
     ('synthesized_image_18', {59: 3, 33: 1, 190: 1, 56: 1, 4: 1, 132: 1}),
     ('synthesized_image_19', {183: 4, 79: 1, 113: 1, 148: 1}),
     ('synthesized_image_20', {195: 1, 159: 1, 72: 1, 65: 1, 125: 1, 120: 2}),
     ('synthesized_image_21', {77: 5, 160: 1, 80: 1, 91: 1, 84: 1, 120: 1, 110: 1, 105: 1}),
     ('synthesized_image_22', {13: 2, 61: 2, 127: 2, 185: 1, 104: 1, 195: 1, 79: 1, 122: 1}),
     ('synthesized_image_23', {12: 1, 30: 1, 102: 1, 44: 2, 91: 1}),
     ('synthesized_image_24', {56: 3, 142: 2, 95: 1, 5: 1, 73: 1, 103: 1, 149: 1, 152: 1}),
     ('synthesized_image_25', {116: 1, 158: 6, 193: 1, 168: 1, 88: 1}),
     ('synthesized_image_26', {55: 3, 112: 3, 20: 1, 32: 3}),
     ('synthesized_image_27', {87: 1, 100: 4, 103: 1, 155: 1, 94: 1, 135: 1, 99: 1, 17: 1, 93: 1}),
     ('synthesized_image_28', {59: 3, 33: 1, 190: 1, 56: 1, 4: 1, 132: 1}),
     ('synthesized_image_29', {183: 4, 79: 1, 113: 1, 148: 1}),
     ('synthesized_image_30', {195: 1, 159: 1, 72: 1, 65: 1, 125: 1, 120: 2}),
     ('synthesized_image_31', {77: 5, 160: 1, 80: 1, 91: 1, 84: 1, 120: 1, 110: 1, 105: 1}),
     ('synthesized_image_32', {13: 2, 61: 2, 127: 2, 185: 1, 104: 1, 195: 1, 79: 1, 122: 1}),
     ('synthesized_image_33', {12: 1, 30: 1, 102: 1, 44: 2, 91: 1}),
     ('synthesized_image_34', {56: 3, 142: 2, 95: 1, 5: 1, 73: 1, 103: 1, 149: 1, 152: 1}),
     ('synthesized_image_35', {116: 1, 158: 6, 193: 1, 168: 1, 88: 1}),
     ('synthesized_image_36', {55: 3, 112: 3, 20: 1, 32: 3}),
     ('synthesized_image_37', {87: 1, 100: 4, 103: 1, 155: 1, 94: 1, 135: 1, 99: 1, 17: 1, 93: 1}),
     ('synthesized_image_38', {59: 3, 33: 1, 190: 1, 56: 1, 4: 1, 132: 1}),
     ('synthesized_image_39', {183: 4, 79: 1, 113: 1, 148: 1})]

names = ['John', 'Ben', 'Bill', 'Alex', 'Jenny'] * 10
image_left = len(x)
y = iter(x)
jobs = {}
m = multiprocessing.Manager()
lock = m.Lock()
MAX_JOBS_IN_QUEUE = 3
with ThreadPoolExecutor(max_workers=5) as executor:
    while image_left:
        for image_id, num_per_category in y:
            # image_id = next(iter(strat.keys()))
            # num_per_category = next(iter(strat.values()))
            job = executor.submit(dummy_def, image_id, num_per_category, lock=lock)
            jobs[job] = image_id
            if len(jobs) > MAX_JOBS_IN_QUEUE:
                break  # limit the job submission for now job
        for job in as_completed(jobs):
            # image_cnt_res = jobs[job]
            # print("done {} image".format(image_cnt_res))
            del jobs[job]
            image_left -= 1
            break
