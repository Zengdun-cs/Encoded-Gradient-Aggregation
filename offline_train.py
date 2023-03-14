import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import argparse
from scipy import stats

from fedlab.utils.functional import AverageMeter, get_best_gpu, setup_seed
from fedlab.utils.logger import Logger

from model import SymmetryEncoder, SymmetryDecoder
from utils import create_dataset, create_seq_dataset
from coder import PipelineAutoEncoder

import time
import numpy as np

def evaluate(model, testloader, args):
    result = []
    prefer = []
    low_mistake = []
    high_mistake = []
    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data = data.cuda(args.gpu)
            target = target.cuda(args.gpu)
            pred = model(data)

            diff = (pred - target).view(-1)

            result.append(diff)
            prefer.append(pred)

    result = torch.cat(result, dim=0).cpu()
    prefer = torch.cat(prefer, dim=0).cpu()
    return result, prefer

if __name__ == "__main__":
    # python offline_train.py -bs 1024 -epoch 1000 -train_size 10000 -test_size 1000 -m 10 -s 32
    parser = argparse.ArgumentParser(description='Aggregation Network')

    parser.add_argument("-lr", type=float, default=0.00001)
    parser.add_argument("-bs", type=int)
    parser.add_argument("-epoch", type=int)
    parser.add_argument("-optim", type=str, default="adam")
    parser.add_argument("-encoder_size", type=int, default=3)
    parser.add_argument("-decoder_size", type=int, default=3)

    parser.add_argument("-m", type=int)
    parser.add_argument("-s", type=int)
    parser.add_argument("--b", type=int, default=512)
    parser.add_argument("--h", type=int, default=128)

    parser.add_argument("-cuda", type=bool, default=True)
    parser.add_argument("-gpu", type=str, default="0,1,2,3")

    parser.add_argument("-freq", type=int, default=5)

    parser.add_argument("-train_size", type=int)
    parser.add_argument("-test_size", type=int)

    parser.add_argument("-spr", type=float, default=0.5) # sparse ratio
    parser.add_argument("-test_spr", type=float, default=0.5) # sparse ratio of test set

    parser.add_argument("-log", type=str, default="logs")
    
    args = parser.parse_args()

    # record
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu = get_best_gpu()
    timestamp = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    logname = "{}-{}-{}-{}-train{}-test{}-coder{}{}-{}".format(args.b, args.h, args.m, args.s, args.spr, args.test_spr, args.encoder_size, args.decoder_size, timestamp)
    root = os.path.join("./", args.log, "LOG-" + logname)
    os.makedirs(root)
    os.mkdir(os.path.join(root, "best_ckpt"))
    print(root)

    train_log = Logger("train_loss",
                       os.path.join(root, "train_loss" + timestamp + ".txt"))

    test_log = Logger("test_loss",
                      os.path.join(root, "test_loss" + timestamp + ".txt"))

    best_std, best_epoch = 1e5, 0

    std = []
    seed = int(time.time())
    setup_seed(seed)

    # settings
    encoder = SymmetryEncoder(num_block=args.encoder_size,
                              in_dim=args.b,
                              out_dim=args.h)

    decoder_in_dim = args.h
 
    decoder = SymmetryDecoder(num_block=args.decoder_size,
                              in_dim=decoder_in_dim,
                              out_dim=args.b)

    model = PipelineAutoEncoder(encoder, decoder, args.s).cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l2loss = torch.nn.MSELoss(reduction="mean")

    test_seq_dataset = create_seq_dataset(args.b, args.s, args.test_size*args.m)
    
    for epoch in range(args.epoch):
        model.train()
        loss_ = AverageMeter()
        begin_time = time.time()
        seq_dataset = create_seq_dataset(args.b, args.s, args.train_size*args.m)
        
        shuffle_ = 10
        for _ in range(shuffle_):
            trainset = create_dataset(seq_dataset, args.m, True, args.spr)
            trainloader = DataLoader(trainset, batch_size=args.bs)
            for step, (data, target) in enumerate(trainloader):
                if args.cuda is True:
                    data = data.cuda(args.gpu)
                    target = target.cuda(args.gpu)

                output = model(data)

                loss = l2loss(output, target) # + noise

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_.update(loss.detach().item())

        testset = create_dataset(test_seq_dataset, args.m, sparse=True, ratio=args.test_spr)
        testloader = DataLoader(testset, batch_size=1000)

        result, prefer = evaluate(model, testloader, args)
        result_np = result.numpy()

        norm_res = (result_np - np.mean(result_np)) / np.std(result_np)
        statistic, p = stats.kstest(norm_res, "norm")
        train_log.info("KS test: stat {}, p_value {}".format(statistic, p))

        std_var = np.std(result_np)
        std.append(std_var)

        log_str = "encoder size {}; decoder size {};\nraw dim {} -> hidden dim {}; Client num {};\ns {} \n".format(
            args.encoder_size, args.decoder_size, args.b,
            args.h, args.m, args.s)

        log_str += str(std)

        with open(os.path.join(root, "record.txt"), "w") as f:
            f.write(str(args))
            f.write(log_str)

        test_log.info("pred error std_var {}".format(std_var))
    
        if std_var <= best_std:
            test_log.info("best pred error std_var {}".format(std_var))
            best_std = np.mean(result_np)
            best_var = np.var(result_np)
            best_std = np.std(result_np)
            best_epoch = epoch

            ckpt = {
                "encoder": model.encoder.state_dict(),
                "encoder_size": args.encoder_size,
                "decoder": model.decoder.state_dict(),
                "decoder_size": args.decoder_size,
                "b": args.b,
                "h": args.h,
                "s": args.s,
                "m": args.m,
                "performance": std_var,
                "seed": seed
            }
            torch.save(ckpt,
                       os.path.join(root, "best_ckpt", "best_model.ckpt.ptr"))
        train_log.info(
            "Epoch: {}/{},  train loss_sum: {:.4f}, train loss_avg: {:.4f}  time: {:.2f}s"
            .format(epoch + 1, args.epoch, loss_.sum, loss_.avg,
                    time.time() - begin_time))
