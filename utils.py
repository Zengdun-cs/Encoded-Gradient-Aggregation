import torch
from torch.utils.data import Dataset, DataLoader
from fedlab.utils.functional import AverageMeter
import seaborn as sns
import random


def have_nan(tensor):
    return torch.any(torch.isnan(tensor))

def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy."""
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg

def obs_values(tensor, block, root):
    tensor = tensor.abs()
    values = [
        torch.sum(tensor[i:i + block]).item()
        for i in range(0, len(tensor), block)
    ]
    data = {"index": [i for i in range(len(values))], "value": values}
    plot = sns.scatterplot(data=data, x="index", y="value")
    fig = plot.get_figure()
    fig.savefig(root)
    plot.cla()


def draw_hist(data, root):
    plot = sns.histplot(data)
    fig = plot.get_figure()
    fig.savefig(root)
    plot.cla()


def draw_dist(data, root):
    plot = sns.distplot(data)
    fig = plot.get_figure()
    fig.savefig(root)
    plot.cla()


def pos_neg(tensor):
    zero_ = torch.zeros_like(tensor)
    pos_t = tensor.where(tensor > 0, zero_)
    neg_t = tensor.where(tensor < 0, zero_).abs()

    return pos_t, neg_t


def tensor_split(tensor, n, fill_last=True):

    tensor_list = [tensor[i:i + n] for i in range(0, tensor.shape[0], n)]

    if fill_last is True and tensor_list[-1].shape[0] != n:
        len = tensor_list[-1].shape[0]
        zeros = torch.zeros(size=(n - len, ))
        tensor_list[-1] = torch.cat([tensor_list[-1], zeros])

    return torch.stack(tensor_list)


def code_parameters(coder, parameters, block_size):
    splited_parameters = tensor_split(parameters, block_size)
    stacked_parameters = torch.stack(splited_parameters).cuda()
    output = coder(stacked_parameters)
    return output

# pretrain dataset
class ListDataset(Dataset):
    """ Basic Dataset Class """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.targets[idx]
        return data, label


def sparsify(tensor, alpha):
    if alpha == 0:
        return tensor

    shape = tensor.shape
    sparsified_ = tensor.view(-1)
    idx = torch.randperm(sparsified_.shape[0])

    zero_idx = idx[0:int(idx.shape[0] * alpha)]
    sparsified_[zero_idx] = 0
    tensor = sparsified_.view(shape)

    return tensor


def gen_sp_sample(client_num, length, sparse, interval):
    tensor = torch.randint(low=0,
                           high=interval + 1,
                           size=(client_num, length),
                           dtype=torch.float32)
    #tensor = sparsify(tensor, sparse)
    mean = torch.mean(tensor, dim=0)
    return tensor, mean


def create_seq_dataset(domain, interval, dataset_size):
    dataset = [
        torch.randint(low=-interval,
                      high=interval + 1,
                      size=(domain, ),
                      dtype=torch.float32) for _ in range(dataset_size)
    ]
    return dataset


def create_dataset(seq_dataset, client_num, sparse=True, ratio=0.5):
    random.shuffle(seq_dataset)
    tensor_list = [
        torch.stack(seq_dataset[i:i + client_num])
        for i in range(0, len(seq_dataset), client_num)
    ]
    for i in range(0, len(seq_dataset), client_num):
        sample = torch.stack(seq_dataset[i:i + client_num])
        if sparse:
            sample = sparsify(sample, ratio)
            # sample *= torch.randint(low=0,
            #                         high=2,
            #                         size=sample.shape,
            #                         dtype=torch.float32)
        tensor_list.append(sample)
    label = [torch.mean(tensor, dim=0) for tensor in tensor_list]
    dataset = ListDataset(tensor_list, label)
    return dataset