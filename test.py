import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2  # opencv默认读取形式为BGR，plt默认读取形式为RGB
from model import UNet


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((480, 480)),
                                    transforms.ToTensor()
                                    ])

    # 定义测试集路径
    test_path = os.listdir(r"C:\Users\86188\PycharmProjects\Unet\DRIVE\test\images\\")  # 这里打开的文件格式是该路径下的"文件名.后缀"
    # print(test_path)

    # 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load("net_parameter.pth", map_location=device))
    model.to(device)

    # 定义阈值
    threshold = 0.5

    # 测试
    model.eval()
    with torch.no_grad():
        for path in test_path:
            if path.find("_res") != -1:
                continue
            save_pre_path = r"C:\Users\86188\PycharmProjects\Unet\DRIVE\test\images\\" + path.split('.')[-2] + "_res.png"
            img = Image.open(r"C:\Users\86188\PycharmProjects\Unet\DRIVE\test\images\\" + path)
            width, height = img.size[0], img.size[1]  # 保存未transform前图片大小，后续进行还原
            img = transform(img)
            img = img.to(device)   # 先改为tensor后用pytorch中的to方法把其放到运行设备上
            img = torch.unsqueeze(img, dim=0)  # 扩展维度

            pred = model(img).clamp(0.0, 1.0)  # 模型预测

            pred = torch.squeeze(pred)  # 将(batch、channel)维度去掉
            pred = np.array(pred.data.cpu(), dtype=np.uint8)  # 保存图片：转为cpu处理,再转为numpy的ndarray（numpy只能在cpu上运行），由于numpy默认数据类型为float64，需更改为uint8(0~255)

            # 对pred[0.0, 1.0]内结果进行二值化
            pred[pred >= threshold] = 255
            pred[pred < threshold] = 0

            pred = cv2.resize(pred, (width, height), cv2.INTER_CUBIC)  # 还原图像的size
            cv2.imwrite(save_pre_path, pred)  # 保存图片
