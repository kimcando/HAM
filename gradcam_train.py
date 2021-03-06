# version for 32x32
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# logging libraries
import wandb

# custom libraries
from arguments import get_args
from dataset import HAM10000, preprocess_df
from model import resnet8_gn, resnet18_modify, BaseCNN, CNN, GradCamModel

# visualize
import torchvision
import matplotlib.pyplot as plt
def xavier_nets(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except:
            pass



def init_nets(args):
    nets = {net_i: None for net_i in range(args.n_parties)}

    for net_i in range(args.n_parties):
        net = GradCamModel(args)
        """
        if args.model == "resnet8_gn":
            net = resnet8_gn(num_classes=args.num_classes)
            if args.init_type == "xavier":
                print('reinitilize with xavier')
                net.apply(xavier_nets)
            else:
                print('sustain he init')
        elif args.model == "resnet18":
            net = resnet18_modify(num_classes=args.num_classes, freeze=args.freeze, bn_freeze = args.bn_freeze, use_pretrained=args.pretrained)

        elif args.model == "basemodel":
            net = BaseCNN(num_classes=args.num_classes)
        elif args.model =='cnn':
            net = CNN(num_classes=args.num_classes)
        else:
            raise NotImplementedError
        """
        nets[net_i] = net
        model_meta_data = []
        layer_type = []

        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
    return nets, model_meta_data, layer_type

def train_net(net_id, net,
              optimizer, scheduler,
              train_dataloader, epochs, device="cuda",
              do_pre=False):
    global norm_mean, norm_std
    criterion = nn.CrossEntropyLoss().to(device)

    batch_cnt = 0
    data_cnt = 0

    loss_val = 0
    acc_val = 0

    net.train()
    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            target = target.long()
            # https://ichi.pro/ko/pytorchui-gradcam-135721179052517

            ## output ??????
            out, selected = net(x) # selected shape: torch.Size([64, 64, 8, 8])
            selected = selected.detach().cpu()

            loss = criterion(out, target)

            _, pred = torch.max(out, 1)
            correct = torch.sum(pred == target)
            loss.backward()
            if batch_idx <20:
                img_idx = 10
                grads = net.get_act_grads().detach().cpu()
                pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
                for i in range(selected.shape[1]):
                    selected[:, i, :, :] += pooled_grads[i]
                heatmap_j = torch.mean(selected, dim=1).squeeze()
                heatmap_j_max = heatmap_j.max(axis=0)[0]
                heatmap_j /= heatmap_j_max

                # ????????? ?????? -> baseline code??? ??? ??? ????????? ??? ??????
                unorm = UnNormalize(mean=norm_mean, std = norm_std)
                resize = torchvision.transforms.Resize((32, 32))
                heatmap_j = resize(heatmap_j)
                # heatmap_j = resize(heatmap_j, (32, 32), preserve_range=True)
                trs = unorm(x[img_idx])*255
                img=trs.detach().cpu().permute(1,2,0).numpy().astype('uint8')

                fig=plt.figure(figsize=(8,8))
                plt.imshow(img)
                plt.imshow(heatmap_j[img_idx],cmap=plt.cm.jet, alpha=0.2, interpolation='nearest')
                plt.savefig(f'./imgs/{img_idx}.png')

            optimizer.step()

            batch_cnt += 1
            data_cnt += target.shape[0]

            loss_val += loss.detach().item()
            acc_val += correct


        # logger.info(
        #     f'Learning rate for {net_id} client: {scheduler.optimizer.param_groups[0]["lr"]}, {scheduler.get_last_lr()}')
        #
        # logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    total_acc = acc_val/data_cnt
    total_loss = loss_val/batch_cnt
    return total_acc, total_loss, scheduler.optimizer.param_groups[0]["lr"]

def test_net(net_id, net,
              test_dataloader, epochs, device="cuda",
              do_pre=False):

    criterion = nn.CrossEntropyLoss().to(device)

    batch_cnt = 0
    data_cnt = 0

    loss_val = 0
    acc_val = 0
    net.eval()
    with torch.no_grad():
        for epoch in range(epochs):
            for batch_idx, (x, target) in enumerate(test_dataloader):
                x, target = x.to(device), target.to(device)

                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                _, pred = torch.max(out, 1)
                correct = torch.sum(pred == target)

                batch_cnt += 1
                data_cnt += target.shape[0]

                loss_val += loss.detach().item()
                acc_val += correct

        # logger.info(
        #     f'Learning rate for {net_id} client: {scheduler.optimizer.param_groups[0]["lr"]}, {scheduler.get_last_lr()}')
        #
        # logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    total_acc = acc_val/data_cnt
    total_loss = loss_val/batch_cnt
    return total_acc, total_loss

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

if __name__=='__main__':

    args = get_args()
    if args.pretrained:
        pretrain='pretrain'
    else:
        pretrain = 'scratch'
    if args.wandb_log:
        wandb.init(project='single_model',
                   name=f'{args.exp_name}_{args.model}_{args.lr}_{pretrain}',
                   entity='feddu')
        wandb.config.update(args)

    #-- model
    nets, local_model_meta_data, layer_type = init_nets(args)
    net_id = 0
    net = nets[net_id]
    net.to(args.device)

    # -- transform
    # precomputed values. refer to jupyter notebook
    norm_mean, norm_std = [0.7630318, 0.5456445, 0.5700395], [0.1409281, 0.15261307, 0.16997099]
    # train_transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
    #                                         transforms.ToTensor(),
    #                                       transforms.Normalize(norm_mean, norm_std)])
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
    # train_transform = transforms.Compose([transforms.ToTensor(),
    #                                       ])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    # -- dataset
    df_train, df_val = preprocess_df(args)
    # Define the training set using the table train_df and using our defined transitions (train_transform)
    # training_set = HAM10000(df_train, transform=train_transform)
    training_set = HAM10000(df_train, mode='train', transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Same for the validation set:
    # validation_set = HAM10000(df_val, transform=val_transform)
    validation_set = HAM10000(df_val,mode='eval', transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=4)

    # -- optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                          momentum=args.rho,
                          weight_decay=args.reg, nesterov=args.nesterov)
    mile_step = list(map(int, args.lr_decay_stepsize))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_step,
                                               gamma=args.lr_decay_gamma)

    for round in tqdm(range(args.comm_round)):
        train_acc, train_loss, train_lr = train_net(net_id, net,
                  optimizer, scheduler,
                  train_dataloader=train_loader,
                  epochs=args.epochs)
        if torch.isnan(torch.tensor(train_loss)):
            break
        print(f' Training at {round} : loss: {train_loss:.6f}, acc:{train_acc:.6%}')
        # test_acc, test_loss = test_net(net_id, net,
        #                                         test_dataloader=val_loader,
        #                                         epochs=args.epochs)
        # print(f' Test at {round} : loss: {test_loss:.6f}, acc:{test_acc:2.6%}')

        if args.wandb_log:
            wandb.log({
                'Train/acc': train_acc,
                'Train/loss': train_loss,
                'Train/lr': train_lr,
                'Test/acc': test_acc,
                'Test/loss': test_loss
            })
