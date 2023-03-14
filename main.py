import os
import argparse
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import torch
import time

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu, setup_seed
from fedlab.utils.logger import Logger

from model import SymmetryEncoder, SymmetryDecoder, CNN_MNIST
from coder import GradDecoder, GradEncoder

# python main.py -ckpt LOG-512-128-10-32-train0.5-test0.5-coder33-03-14-16-22-28 -ll 0.01 -bs 128 -ep 5 -com_round 100
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("-com_round", type=int, required=True)

parser.add_argument("-ckpt", type=str, required=True)

# local training
parser.add_argument("-bs", type=int)
parser.add_argument("-ll", type=float) # learning rate local
parser.add_argument("-ep", type=int)

# global
parser.add_argument("-lg", type=float, default=1) # learning rate global
parser.add_argument("-alg", type=str, default="fedavg")

parser.add_argument("-n_scale", type=float, default=1)
args = parser.parse_args()


gpu = get_best_gpu()
model = CNN_MNIST()

# setup
seed = int(time.time())
setup_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ckpt = torch.load(
    "./logs/" + args.ckpt + "/best_ckpt/best_model.ckpt.ptr", map_location="cpu"
)

# please see the paper for definition
b = ckpt["b"]
h = ckpt["h"]
m = ckpt["m"]
s = ckpt["s"]

# encoder-decoder architechture
encoder_size = ckpt["encoder_size"]
decoder_size = ckpt["decoder_size"]

encoder = SymmetryEncoder(encoder_size, in_dim=b, out_dim=h)
encoder.load_state_dict(ckpt["encoder"])

decoder = SymmetryDecoder(decoder_size, in_dim=h, out_dim=b)
decoder.load_state_dict(ckpt["decoder"])

# logs
timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
root = "./logs/" + args.ckpt + "/runs/"
# os.makedirs(root)
rec_log = Logger("record", os.path.join(root, "EGA" + timestamp + ".txt"))
debug_log = Logger("debug", os.path.join(root, "DEBUG" + timestamp + ".txt"))

# remove
ckpt.pop("encoder")
ckpt.pop("decoder")

encoder = GradEncoder(encoder, b)
decoder = GradDecoder(decoder, h)

# datasets
mnist = PathologicalMNIST("./data/mnist/", "./data/fed_mnist/", num_clients=100, shards=1000)
mnist.preprocess()
testset = torchvision.datasets.MNIST(root='./data/mnist/',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=1024)

# fedlab setup
trainer = SGDSerialClientTrainer(model, num_clients=mnist.num_clients)
trainer.setup_dataset(mnist)
trainer.setup_optim(args.ep, args.bs, args.ll)

handler = SyncServerHandler(model, global_round=args.com_round, sample_ratio=float(m/mnist.num_clients))
handler.num_clients = trainer.num_clients

rec_loss, rec_acc = [], []
while handler.if_stop is False:
    sampled_clients = handler.sample_clients()

    broadcast = handler.downlink_package

    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    gradient_list = [torch.sub(handler.model_parameters, ele[0]) for ele in uploads]
    norm = max([torch.norm(gradient, p=2, dim=0).item() for gradient in gradient_list])

    # encoding locally
    encoded_signature = [encoder(grad, norm, s) for grad in gradient_list]
    encoded_gradients = [endoced[1] * s for endoced in encoded_signature]

    # aggregate privately
    aggregated = torch.mean(torch.stack(encoded_gradients), dim=0)

    # decode globally
    decoded_signature = decoder(aggregated, norm, s) # quantized, decoded
    decoded_gradient = decoded_signature[1][0:handler.model_parameters.numel()] # align model paramters

    aggregated_parameters = handler.model_parameters - args.lg * decoded_gradient
    SerializationTool.deserialize_model(handler._model, aggregated_parameters)

    handler.round += 1
    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), testloader)
    rec_log.info("Round: {},loss: {:.4f}, acc: {:.2f}".format(handler.round, loss, acc))
    
    # debug
    raw_gradient = torch.mean(torch.stack(gradient_list), dim=0).cpu()
    diff = (raw_gradient - decoded_gradient).abs()
    debug_log.info(
            "Debug - diff - sum {}, mean {}, max {}, min {}, std {}".format(
                diff.sum(), diff.mean(), diff.max(), diff.min(), diff.std()
            )
        )
