import os
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import csv
import torch.nn as nn
from my_dataset import MyDataSet
from dataset import RafDataset
# from model_se import mobile_vit_xx_small as create_model
from model_fast import Creatmodel
from fasternet import fasternet_s as create_model
from utils import read_split_data, train_one_epoch, evaluate
from sklearn.metrics import f1_score, recall_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0004)
from torchvision.models.resnet import resnet50
# 数据集所在根目录
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
parser.add_argument('--data-path', type=str,
                    default=r"E:\sunrui\datasets\AF7")
parser.add_argument('--workers', type=int, default=4, help='number of workers')

parser.add_argument('--datasets', type=str,
                    default='AF7')
# 预训练权重路径，如果不想载入就设置为空字符
parser.add_argument('--weights', type=str, default='',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

# custom style of time
time_format = '%Y_%m_%d_%X'
parser.add_argument('--current_time', default=str(time.strftime(time_format).replace(":", "_")), type=str,
                    help='Generate current time when project runs to make some relative folders')

opt = parser.parse_args()
def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        acc = correct.float().sum().mul_(1.0 / batch_size)
    return acc, pred
def save_checkpoint(epoch, model, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join("./savemodel",tag))
def calc_metrics(y_pred, y_true, y_scores):
    metrics = {}
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()
    y_scores = torch.cat(y_scores).cpu().numpy()
    classes = unique_labels(y_true, y_pred)

    # recall score
    metrics['rec'] = recall_score(y_true, y_pred, average='macro')

    # f1 score
    f1_scores = f1_score(y_true, y_pred, average=None, labels=unique_labels(y_pred))
    metrics['f1'] = f1_scores.sum() / classes.shape[0]

    # AUC PR
    Y = label_binarize(y_true, classes=classes.astype(int).tolist())
    metrics['aucpr'] = average_precision_score(Y, y_scores, average='macro')

    # AUC ROC
    metrics['aucroc'] = roc_auc_score(Y, y_scores, average='macro')

    return metrics
def main(args):
    with open('log_' + str(opt.data_path[-3:]) + '.txt', 'a') as f:
        f.write(
            'lr: {}   dataset: {}'.format(
                opt.lr, opt.data_path[-3:]) + '\n')
    print(torch.cuda.is_available())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    if os.path.exists("./runs") is False:
        os.makedirs("./runs")

    save_folder_path = f"./runs/{opt.current_time}"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    tb_writer = SummaryWriter(log_dir=save_folder_path, flush_secs=10)

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
    #                                  # transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                                  # transforms.RandomErasing(scale=(0.02, 0.25))
    #                             ]),
    #     "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                transforms.CenterCrop(img_size),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])
    # train_set = datasets.ImageFolder(
    #     root=os.path.join(opt.data_path, 'train'),
    #     transform=data_transform["train"]
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=opt.batch_size, shuffle=True,
    #     num_workers=4, pin_memory=True)
    #
    # # validation set
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=datasets.ImageFolder(
    #         root=os.path.join(opt.data_path, 'val'),
    #         transform=data_transform["val"]
    #     ),
    #     batch_size=opt.batch_size, shuffle=False,
    #     num_workers=4, pin_memory=True
    # )
    data_transform = {
        'train': transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(12),
                                     transforms.RandomCrop((224, 224)),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     transforms.RandomErasing(scale=(0.02, 0.25))]),

        'val': transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.CenterCrop((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]),
    }
    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])
    train_set = datasets.ImageFolder(
        root=os.path.join(opt.data_path, 'train'),
        transform=data_transform["train"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    # validation set
    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(opt.data_path, 'val'),
            transform=data_transform["val"]
        ),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    batch_size = opt.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 4
    print('Using {} dataloader workers every process'.format(nw))


    # model = create_model(num_classes=args.num_classes).to(device)
    model = Creatmodel(args,pretrained=False).to(device)

    print(model)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10,  # init_epoch to change lr
                                                                     T_mult=2,  # times
                                                                     eta_min=0,  # min of lr
                                                                     last_epoch=-1,  # default=-1
                                                                     )

    best_acc = 0.

    for epoch in range(args.epochs):
        y_pred = []
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     dataset = str(opt.data_path[-3:]) )

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        with open('log_'+str(args.datasets)+'.txt', 'a') as f:
            f.write('epoch: {:.0f}   train_acc: {:.4f}  train_loss: {:.4f} test_acc: {:.4f}  test_loss: {:.4f}'.format(int(epoch),train_acc, train_loss, val_acc,val_loss)+'\n')

        # 调整余弦退火的学习率
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_folder_path}/best_model.pth")
            with open('y_pred.csv', "w", newline='') as f:
                y_pred = list(map(lambda x: [x], y_pred))
                writer = csv.writer(f)
                for row in y_pred:
                    writer.writerow(row)
        # # with open('log_'+str(opt.data_path[-3:])+'.txt', 'a') as f:
        # #     f.write('epoch: {:.0f}   train_acc: {:.4f}  train_loss: {:.4f} test_acc: {:.4f}  test_loss: {:.4f} best_acc: {:.4f}'.format(int(epoch),train_acc, train_loss, val_acc,val_loss,best_acc)+'\n')

        torch.save(model.state_dict(), f"{save_folder_path}/latest_model.pth")


if __name__ == '__main__':


    main(opt)

