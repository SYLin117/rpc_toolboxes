# import json
# import os
#
# demo_json_path = os.path.join('D:/datasets/retail_product_checkout', 'instances_val2019.json')
# with open(demo_json_path) as fid:
#     demo_data = json.load(fid)
#
# new_json_path = os.path.join('D:/datasets/retail_product_checkout', 'instances_val2019_removed.json')
# with open(new_json_path) as fid:
#     new_data = json.load(fid)
#
# new_data['info'] = demo_data['info']
# new_data['licenses'] = demo_data['licenses']
# new_data['categories'] = demo_data['categories']
# new_data['__raw_Chinese_name_df'] = demo_data['__raw_Chinese_name_df']
#
# with open(new_json_path, 'w') as fid:
#     json.dump(new_data, fid)
# ## ===================================================================
# # new_json_path = os.path.join('D:/datasets/retail_product_checkout', 'instances_val2019_removed.json')
# # with open(new_json_path) as fid:
# #     new_data = json.load(fid)
# # save_path = 'D:/datasets/retail_product_checkout/val2019_removed'
# # for image in new_data['images']:
#
