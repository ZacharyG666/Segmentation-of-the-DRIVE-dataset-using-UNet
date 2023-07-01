import torch
from torch import nn
import torch.nn.functional as F

# 连续俩次卷积 由于padding = kernel_size // 2 故连续俩次卷积后输出特征图大小不变
# 堆叠小的卷积核所需的参数更少一些，并且卷积过程越多，特征提取越细致，加入的非线性变换也越多，还不会增大权重参数个数，这就是VGG网络的出发点“用小的卷积核完成特征提取操作”
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 下采样
class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 上采样
class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        # self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)         # 双线性插值
        self.upsampling = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)   # 转置卷积
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
       x1 = self.upsampling(x1)

       # 确保任意size图像输入
       diffY = torch.tensor(x2.size()[2] - x1.size()[2])  # 计算x2、x1高度间差异
       diffX = torch.tensor(x2.size()[3] - x1.size()[3])  # 计算x2、x1宽度间差异

       # 为了确保俩个tensor的size相同，使用F.pad进行零填充
       x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
       x = torch.cat([x2, x1], dim=1)  # NCHW 从channel拼接
       x = self.conv(x)
       return x

# Unet应用：分割，超分辨，去噪
class UNet(nn.Module):
    def __init__(self, in_channel=3, num_channel=1):
        super(UNet, self).__init__()

        n = 64
        filters = [n, n * 2, n * 4, n * 8, n * 16]   # 64 128 256 512 1024

        # 输入
        self.in_conv = nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1, bias=False)  # 保证输入卷积层后图像大小不变
        # 下采样
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.down4 = Down(filters[3], filters[4])

        # 上采样
        self.up1 = Up(filters[4], filters[3])
        self.up2 = Up(filters[3], filters[2])
        self.up3 = Up(filters[2], filters[1])
        self.up4 = Up(filters[1], filters[0])

        # 输出
        # 1×1卷积核所做的卷积又称逐点卷积，与传统卷积核所做的卷积不同，其不具备感受野概念，它只是对输入特征图的每个像素点进行线性变换和非线性激活
        # 1×1卷积核所做的卷积通常用来调整通道数和特征图的维度，通过改变卷积核的个数、通道数、权重来实现
        # 在深度学习模型中，1×1卷积核通常用于增加模型的非线性表达能力和降低计算复杂度
        self.out_conv = nn.Conv2d(filters[0], num_channel, kernel_size=1)

    # 在 PyTorch 中，model() 和 model.forward() 这两种方式是等价的，它们实际上都是调用 nn.Module 类中的 __call__ 方法来实现的。
    # 通过 model() 调用神经网络模型的 forward 函数时，实际上是调用了 nn.Module 类中的 __call__ 方法。
    # 在 __call__ 方法中，会调用 forward 方法来实现前向传播计算。
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x



# if __name__ == "__main__":
#     model = UNet()
#     a = torch.rand(size=(1, 3, 256, 256))
#     preds = model(a)
#     print(model)







