"""Batch Sampler
"""

import math
import numpy as np

from torch.utils.data.sampler import Sampler


class BatchSampler(Sampler):
    """Sampler used in dataloader. Method __iter__ should output \
            the indices each time when it's called
    """

    def __init__(self, dataset, n_classes, n_num):
        super(BatchSampler, self).__init__(dataset)
        self.n_classes = n_classes
        self.n_num = n_num
        self.batch_size = self.n_classes * self.n_num
        self.targets_uniq = dataset.targets_uniq
        self.targets = np.array(dataset.targets)
        self.dataset = dataset
        self.target_img_dict = dataset.target_img_dict
        self.len = len(dataset)
        self.iter_num = len(self.targets_uniq) // self.n_classes
        self.repeat = math.ceil(self.len / self.batch_size)

    def __iter__(self):
        # for _ in range(self.repeat):
        curr_p = 0
        np.random.shuffle(self.targets_uniq)
        for k, v in self.target_img_dict.items():
            np.random.shuffle(self.target_img_dict[k])

        for i in range(self.iter_num):
            target_batch = self.targets_uniq[curr_p: curr_p + self.n_classes]
            curr_p += self.n_classes
            idx = []
            for target in target_batch:
                if len(self.target_img_dict[target]) > self.n_num:
                    idx_smp = np.random.choice(self.target_img_dict[target], self.n_num, replace=False)
                else:
                    idx_smp = np.random.choice(self.target_img_dict[target], self.n_num, replace=True)
                idx.extend(idx_smp.tolist())
            yield idx

    def __len__(self):
        return self.iter_num
