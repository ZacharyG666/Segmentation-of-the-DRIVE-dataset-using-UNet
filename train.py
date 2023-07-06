import torch
import copy
from torch.backends import cudnn
from utlis import AverageMeter
from data import DRIVEDataset
from model import UNet
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import utils as vutils
from tqdm import tqdm


if __name__ == '__main__':
    # 定义超参数
    start_epoch = 0
    num_epoch = 20
    batch_size = 1
    lr = 1e-5

    # ①.网络训练模块
    cudnn.benchmark = True  # 在训练中自动寻找最优算法，以达到快速的训练速度。当模型重复性较重要时，可将该设置关闭
    # 检查当前系统是否有可用的GPU，并将其作为模型的运行设备。如果有可用的GPU，则将设备设置为’cuda:0’；否则，将其设置为CPU。
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet()        # 加载网络
    model.to(device)      # 将网络加载到device上

    # Dataset 提供一种方式获取数据及其label---> ①. __getitem__：如何获取每一个数据及其label ②. __len__：告诉总共有多少的数据
    # Dataloader 为后面的网络提供不同的数据形式

    # ----原训练集----   其中images为输入集、1st_manual 为语义分割结果、mask为显著性目标检测(SOD)结果
    train_path = "./DRIVE/training"
    train_set = DRIVEDataset(train_path)

    # 从原训练集集中抽取80％作训练集、20%作验证集
    train_size = int(0.8 * len(train_set))
    eval_size = len(train_set) - train_size
    train_set, eval_set = random_split(train_set, [train_size, eval_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    # ③.加载损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 保存网络参数
    save_path = "./net_parameter.pth"
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0


    # 训练
    for epoch in range(start_epoch, num_epoch):
        model.train()
        epoch_train_losses = AverageMeter()  # 计算和存储当前epoch中所有批次损失函数的平均值，以便实时了解模型的训练效果

        # total 表示总共迭代次数，是≤len(train_set)的最大的batch_size的倍数
        # 例如，如果len(train_set)为1000，batch_size为128，则len(train_set) - len(train_set) % batch_size将等于896，这是小于或等于1000的最大的128的倍数。因此，进度条将被分成896个步骤，每个步骤表示128个元素的一批。
        with tqdm(total=(len(train_set) - len(train_set) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epoch - 1))

            for data in train_loader:
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)

                vutils.save_image(imgs[0, :, :, :], "input.png")
                vutils.save_image(labels[0, :, :, :], "ground_truth.png")

                optimizer.zero_grad()  # 梯度清零  PyTorch默认将梯度累加到之前的梯度上，因此将优化器的梯度清0，以便计算新的梯度
                preds = model(imgs).clamp(0.0, 1.0)  # 前向传播
                vutils.save_image(preds[0, :, :, :], "output.png")
                loss = criterion(preds, labels)  # 计算损失
                epoch_train_losses.update(loss.item(), len(imgs))  # 通常用于计算损失函数、计算准确率等。需要将张量中的数据转换为标量，才能进行比较和计算。
                loss.backward()  # 反向传播（计算梯度）
                optimizer.step()  # 梯度下降（更新参数，以最小化误差）

                t.set_postfix(loss='{:.6f}'.format(epoch_train_losses.avg))
                t.update(len(imgs))



        # 每个训练周期结束时，使用验证集来评估模型性能，并且根据验证集的误差来调整模型的参数，提高模型的泛化能力
        model.eval()
        epoch_eval_losses = AverageMeter()
        acc = 0.0  # # accumulate accurate number / epoch
        cnt = 0  # 验证集总共有多少张图片

        with torch.no_grad():  # 验证时不需要计算梯度
            for data in eval_loader:
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).clamp(0.0, 1.0)
                eval_losses = criterion(preds, labels)
                epoch_eval_losses.update(eval_losses.item(), len(imgs))
                # .item()将只包含一个元素的张量转换为Python中的标量。这个方法只适用于只包含一个元素的张量，如果张量中有多个元素，将会抛出异常。
                acc += (preds == labels).sum().item() / (labels.size(2) * labels.size(3))
                cnt += labels.size(0)
            eval_acc = acc / cnt   # eval_acc = acc / eval_size
            print("epoch:{}, eval_losses:{:.6f}, epoch_acc:{:.4f}".format(epoch, eval_losses, eval_acc))

        if eval_acc >= best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    print("best epoch:{}, acc:{:.4f}".format(best_epoch, best_acc))
    torch.save(best_weights, save_path)
