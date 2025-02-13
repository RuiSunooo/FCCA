import os
import sys
import json
import pickle
import random
import csv
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from loss import ACLoss
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm as cmm
from matplotlib.colors import LinearSegmentedColormap
from itertools import chain
from sklearn.manifold import TSNE
import umap.plot
import umap
from timm.data.mixup import Mixup
import matplotlib.pyplot
def save_checkpoint(epoch, model, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join("./savemodel",tag))
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add
def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid
def flip_image(image_array):
    return cv2.flip(image_array, 1)
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        # plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        # plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=7)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images1, target = data


        images1, target = images1.to(device, non_blocking=True), Variable(target).to(device, non_blocking=True)
        # print('target:', target.shape)
        images1, labels = mixup_fn(images1, target)
        # print("labels",labels.shape)


        images2 = flip(images1,-1)
        from torchvision.utils import save_image

        # save_image(images1, './check1.jpg')
        from torchvision.utils import save_image

        # save_image(images2, './check2.jpg')
        sample_num += images1.shape[0]

        _,pred1,hm1 = model(images1.to(device))
        _,pred2,hm2 = model(images2.to(device))

        grid_l = generate_flip_grid(7, 7, device)

        flip_loss_l = ACLoss(hm1, hm2, grid_l, pred1)
        pred_classes = torch.max(pred1, dim=1)[1]
        # print('pred.shape', pred1.shape)
        # print('pred_classes.shape',pred_classes.shape)
        # print('arget.shape', target.shape)
        accu_num += torch.eq(pred_classes, target.to(device)).sum()
        # print(pred1.shape,)
        # print(labels.shape)

        loss1 = loss_function(pred1, labels.to(device))



        loss = loss1 + 1 * flip_loss_l
        # loss = loss1

        # print('pred_classes.shape', pred_classes.shape)
        # print('labels.shape', labels.shape)


        # loss = loss_function(pred1, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch,dataset):
    loss_function = torch.nn.CrossEntropyLoss()
    y_pred,y_gt,pred_classes,feat = [],[],[],[]


    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        feature,pred,_ = model(images.to(device))
        # print(type(pred))
        predict_np = np.argmax(pred.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
        labels_np = labels.numpy()  # array([0,5,0,6,2,...],dtype=int64)
        feature = feature.cpu().detach().numpy()

        y_pred.append(predict_np)
        # print(feature.shape)
        # print(feature)

        feat.append(feature)
        y_gt.append(labels_np)
        pred_classes = torch.max(pred, dim=1)[1]
        _, predicts = torch.max(pred, 1)

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # print(pred.shape)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)
    # feat = torch.cat(feat).cpu().data.numpy()
    # print(np.array(feat).shape)
    lab = y_gt
    # if accu_num.item() / sample_num > 85:
    #     save_checkpoint(epoch, model, tag=str(epoch) + '_best_valid_rec.pth')
    #     y_pred = list(np.concatenate(y_pred))
    #     # print(y_pred)
    #
    #     with open('y_pred.csv', "w", newline='') as f:
    #         y_pred = list(map(lambda x: [x], y_pred))
    #         writer = csv.writer(f)
    #         for row in y_pred:
    #             writer.writerow(row)
    # draw_confusion_matrix(label_true=y_gt,  # y_gt=[0,5,1,6,3,...]
    #                       label_pred=y_pred,  # y_pred=[0,5,1,6,3,...]
    #                       label_name=["SUR", "FEA", "DIS", "HAP", "SAD", "ANG", "NEU"],
    #                       normlize=True,
    #                       title="Confusion Matrix on Fer2013",
    #                       pdf_save_path=dataset+'_'+str(epoch)+"_Conf"+'.svg',
    #                       dpi=300)
    # # print("hello")
    # Visualization(feat, lab, epoch,dataset+'_'+str(epoch)+"_feat"+'.svg',300 )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None,
                          dpi=300):
    # print('label_true', label_true)
    # print('label_pred', label_pred)
    # print('label_true',np.array(label_true).shape)
    # print('label_pred',np.array(label_pred).shape)
    y_true, y_pred = [], []
    # y_true = list(chain(y_true.tolist()))
    # y_pred = list(chain(y_pred.tolist()))
    for element in label_pred:
        element = list(chain(element.tolist()))
        # element.tolist()
        y_pred.append(element)
    y_pred = [element for sublist in y_pred for element in sublist]
    for element in label_true:
        element = list(chain(element.tolist()))
        # element.tolist()
        y_true.append(element)
    y_true = [element for sublist in y_true for element in sublist]


    # print('y_true',y_true)

    cm = confusion_matrix(y_true, y_pred)

    if normlize:
        row_sums = np.sum(cm, axis=1)  # 计算每行的和
        cm = cm / row_sums[:, np.newaxis]  # 广播计算每个元素占比
    # cm = cm.T

    # print(cm)
    # print(np.array(cm).shape)
    # viridis = cmm.get_cmap('jet', 256)
    # # 获取颜色序列
    # newcolors = viridis(np.linspace(0, 1, 256))
    # print('newcolors',newcolors)
    # print('newcolor',newcolors.shape)
    # # 自定义颜色
    # white = np.array([1, 1, 1, 1])
    # # 替换原有映射指定范围内的颜色
    # newcolors[:64, :] = white
    # print('newcolors[:2, :]', newcolors[:2, :])
    # print('newcolor[:2, :]', newcolors[:2, :].shape)
    # # 生成新映射
    # newcmp = LinearSegmentedColormap(newcolors)
    color_list = ['#4169E1', '#98FB98', '#FFD700', '#FF4500']

    # 线性补帧，并定义自定义colormap的名字，此处为rain
    my_cmap = LinearSegmentedColormap.from_list('rain', color_list)

    # 注册自定义的cmap，此后可以像使用内置的colormap一样使用自定义的rain
    # cmm.register_cmap(cmap=my_cmap)
    plt.figure()
    plt.imshow(cm, cmap = my_cmap)
    plt.title(title)

    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name)

    ax = plt.subplot(111)  # 设置刻度字体大小

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)  # 设置坐标标签字体大小
    ax.set_xlabel(..., fontsize=20)
    ax.set_ylabel(..., fontsize=20)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            # print('cm[i,j]',cm[i,j])
            # value = cm[i, j]
            value = round(cm.T[i,j], 2)
            value = value * 100
            value = round(value,2)
            # value = float(format('%.2f' % cm[i, j]))
            # print('value',value)

            plt.text(i, j, value, fontsize = 14,verticalalignment='center', horizontalalignment='center', color=color)


    plt.savefig('./saveconf/'+pdf_save_path, format='svg',bbox_inches='tight', dpi=dpi)
    # print('./saveconf/'+pdf_save_path)
    plt.plot()
    plt.close()
    # plt.show()
    # if not pdf_save_path is None:





class Visualization:
    def __init__(self, coordinates, labels, epoch, save_path,dpi):
        self.c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        self.coordinates = coordinates
        self.labels = labels
        self.epoch = epoch
        self.save_path = save_path
        self.dpi = dpi
        self.forward()


    def forward(self):
        coor, lab = [], []
        # y_true = list(chain(y_true.tolist()))
        # y_pred = list(chain(y_pred.tolist()))
        for element in self.coordinates:
            element = list(chain(element.tolist()))
            # element.tolist()
            coor.append(element)
        coor = [element for sublist in coor for element in sublist]

        for element in self.labels:
            element = list(chain(element.tolist()))
            # element.tolist()
            lab.append(element)
        lab = [element for sublist in lab for element in sublist]
        # print(np.array(coor).shape)
        # print(np.array(lab).shape)
        coor = np.array(coor)
        lab = np.array(lab)
        # plt.ion()
        # plt.clf()
        # print('self.coordinates', len(self.coordinates))
        # print('self.labels', len(self.labels))
        # print('self.coordinates',self.coordinates)
        # print('self.labels',self.labels)
        # tsne = umap(n_components=2, init='pca', random_state=0)
        coor = umap.UMAP().fit(coor)

        # coor = np.array(coor)
        # print('coor.shape',coor)
        # print('lab.shape',lab.shape)
        colorlist = ['#95a2ff', '#fa8080', '#ffc076', '#fae768', '#87e885', '#3cb9fc', '#73abf5']

        # fig, ax = umap.plot.plt.subplots(2, 2, figsize=(12, 12))


        p = umap.plot.points(coor, lab,color_key_cmap='Paired',)
        # plt.scatter(coor[:, 0], coor[:, 1], c=lab,s=16)


        # plt.legend(["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"])
        # plt.plot()
        plt.savefig('./savefeat/' + self.save_path, format='svg', bbox_inches='tight', dpi=self.dpi)
        # plt.show(fig)


