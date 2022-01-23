import torch


def to_array(feature_map):
    if feature_map.shape[0] == 1:
        feature_map = feature_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        feature_map = feature_map.permute(0, 2, 3, 1).detach().cpu().numpy()
    return feature_map


def to_tensor(feature_map):
    return torch.as_tensor(feature_map.transpose(0, 3, 1, 2), dtype=torch.float32)


def resave_dataparallel_state_dict(path, new_path):
    state_dict = torch.load(path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    torch.save(new_state_dict, new_path)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)


if __name__ == "__main__":
    resave_dataparallel_state_dict('pretrained\\TRACER-Efficient-0.pth', 'pretrained\\TRACER-Efficient-0-single.pth')
