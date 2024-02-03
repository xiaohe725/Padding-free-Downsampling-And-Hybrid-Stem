import os
import math
import argparse
import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from mobilenetv3_efficientformerv2_downsampling import mobilenet_v3_large
from utils import read_split_data, train_one_epoch, evaluate



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()


    data_transform = {
        "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                                     ]),
                                     
                 
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                                   ])}
                          
    train_dataset = datasets.CIFAR100(root=os.getcwd(), train=True,transform=data_transform["train"],download=False)
    val_dataset =datasets.CIFAR100(root=os.getcwd(),train=False,transform=data_transform["val"],download=False)

    batch_size = args.batch_size
    nw =2 # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = mobilenet_v3_large(num_classes=args.num_classes).to(device)


                
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0
    save_path = "./mobilenet_v3_large_Imagenet.pth"
    save_path2 = "./mobilenet_v3_large_Imagenet_All.pth"
    

    
    for epoch in range(args.epochs):

        optimizer.zero_grad()
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()


        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]

        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if acc > best_acc:
            best_acc = acc
            #torch.save(model.state_dict(), save_path)
            #torch.save(model, save_path2)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--lrf', type=float, default=0.1)


    parser.add_argument('--data-path', type=str,
                        default="")

    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
