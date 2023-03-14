# Encoded Gradients Aggregation against Gradient Leakage for Horizontal Federated Learning

Demo for MNIST experiment on federated learning setting.

## Offline Training

> python offline_train.py -bs 1024 -epoch 1000 -train_size 10000 -test_size 1000 -m 10 -s 32


## Deployment

> python main.py -ckpt {path_ckpt} -ll 0.01 -bs 128 -ep 3 -com_round 100

e.g. python main.py -ckpt LOG-512-128-10-32-train0.5-test0.5-coder33-03-14-16-22-28 -ll 0.01 -bs 128 -ep 3 -com_round 100
## Results

2023-03-14 16:49:12,529 - record - INFO - Round: 1,loss: 0.0022, acc: 0.20   
2023-03-14 16:49:15,502 - record - INFO - Round: 2,loss: 0.0022, acc: 0.19  
2023-03-14 16:49:18,320 - record - INFO - Round: 3,loss: 0.0020, acc: 0.31  
2023-03-14 16:49:21,275 - record - INFO - Round: 4,loss: 0.0017, acc: 0.40  
2023-03-14 16:49:24,245 - record - INFO - Round: 5,loss: 0.0016, acc: 0.41  
...  
2023-03-14 16:54:07,630 - record - INFO - Round: 96,loss: 0.0001, acc: 0.97  
2023-03-14 16:54:10,670 - record - INFO - Round: 97,loss: 0.0001, acc: 0.97  
2023-03-14 16:54:13,848 - record - INFO - Round: 98,loss: 0.0001, acc: 0.98  
2023-03-14 16:54:16,954 - record - INFO - Round: 99,loss: 0.0001, acc: 0.98  
2023-03-14 16:54:20,005 - record - INFO - Round: 100,loss: 0.0001, acc: 0.98  