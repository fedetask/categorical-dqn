import torch
from torch import multiprocessing


class Counter:
    """Shared counter implementation, from https://github.com/Kaixhin/ACER/blob/master/utils.py
    """

    def __init__(self):
        self.val = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()

    def increment(self, v=1):
        with self.lock:
            self.val.value += v

    def value(self):
        with self.lock:
            return self.val.value


def cuda_if(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


def np_to_unsq_tensor(state):
    return torch.from_numpy(state).float().unsqueeze(0)


def squeeze_np(tensor):
    return tensor.squeeze().numpy()
