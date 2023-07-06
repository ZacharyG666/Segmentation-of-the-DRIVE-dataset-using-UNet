import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# torchvision.transforms.ToTensor(): ①.避免不同特征值域差异过大，使网络训练更容易  ②.避免反向传播时梯度消失与梯度爆炸，提高模型的训练效率与性能
data_transform = transforms.Compose([transforms.ToTensor()])
                                    # transforms.Normalize((0.5, ), (0.5, ))])  # 标准化把正态分布转化为标准正态分布


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform=data_transform):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "images")
        self.img_names = os.listdir(self.img_dir)
        self.label_dir = os.path.join(root_dir, "1st_manual")
        self.label_names = os.listdir(self.label_dir)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_item_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_item_path)
        img = self.transform(img)

        label_name = self.label_names[idx]
        label_item_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_item_path)
        label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.img_names)





