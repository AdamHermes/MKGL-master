
import torch


def to_undirected_with_inverse(edge_index, edge_attr, num_relations):
    """
    Mimics TorchDrug's graph.undirected(add_inverse=True)
    1. Flips edges (h, t) -> (t, h)
    2. Creates inverse relations: r -> r + num_relations
    """
    # 1. Create inverse edges (flip source and target)
    edge_index_inv = torch.stack([edge_index[1], edge_index[0]], dim=0)
    
    # 2. Create inverse relations (shift IDs by num_relations)
    # Assuming edge_attr contains relation IDs
    edge_attr_inv = edge_attr + num_relations
    
    # 3. Concatenate original and inverse
    new_edge_index = torch.cat([edge_index, edge_index_inv], dim=1)
    new_edge_attr = torch.cat([edge_attr, edge_attr_inv], dim=0)
    
    return new_edge_index, new_edge_attr
def multikey_argsort(inputs, descending=False, break_tie=False):
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def bincount(input, minlength=0):
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted = (input.diff() >= 0).all()
    if sorted:
        if minlength == 0:
            minlength = input.max() + 1
        range = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range, input)
        return index.diff()

    return input.bincount(minlength=minlength)


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = order[range]

    return input[index], index


allow_materialization = False


class VirtualTensor(object):

    def __init__(self, keys=None, values=None, index=None, input=None, shape=None, dtype=None, device=None):
        if shape is None:
            shape = index.shape + input.shape[1:]
        if index is None:
            index = torch.zeros(*shape[:1], dtype=torch.long, device=device)
        if input is None:
            input = torch.empty(1, *shape[1:], dtype=dtype, device=device)
        if keys is None:
            keys = torch.empty(0, dtype=torch.long, device=device)
        if values is None:
            values = torch.empty(0, *shape[1:], dtype=dtype, device=device)

        self.keys = keys
        self.values = values
        self.index = index
        self.input = input

    @classmethod
    def zeros(cls, *shape, dtype=None, device=None):
        input = torch.zeros(1, *shape[1:], dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def full(cls, shape, value, dtype=None, device=None):
        input = torch.full((1,) + shape[1:], value, dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def gather(cls, input, index):
        return cls(index=index, input=input, dtype=input.dtype, device=input.device)

    def clone(self):
        return VirtualTensor(self.keys.clone(), self.values.clone(), self.index.clone(), self.input.clone())

    @property
    def shape(self):
        return self.index.shape + self.input.shape[1:]

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        keys = indexes[0]
        # print(keys.numel(), keys.max(), len(self.index), keys.min())
        assert keys.numel() == 0 or (keys.max() < len(self.index) and keys.min() >= 0)
        values = self.input[(self.index[keys],) + indexes[1:]]
        if len(self.keys) > 0:
            index = torch.bucketize(keys, self.keys)
            index = index.clamp(max=len(self.keys) - 1)
            indexes = (index,) + indexes[1:]
            found = keys == self.keys[index]
            indexes = tuple(index[found] for index in indexes)
            values[found] = self.values[indexes]
        return values

    def __setitem__(self, keys, values):
        new_keys, inverse = torch.cat(
            [self.keys, keys]).unique(return_inverse=True)
        new_values = torch.zeros(
            len(new_keys), *self.shape[1:], dtype=self.dtype, device=self.device)
        new_values[inverse[:len(self.keys)]] = self.values
        new_values[inverse[len(self.keys):]] = values
        self.keys = new_keys
        self.values = new_values

    def __len__(self):
        return self.shape[0]
    
