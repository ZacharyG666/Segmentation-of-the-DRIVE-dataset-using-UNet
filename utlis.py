# AverageMeter（平均计量器）是一个常用的工具类，用于计算一个指标的平均值和当前值。
# 它通常用于评估训练或测试过程中的性能指标，例如损失函数、准确率或其他指标。
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.   # 目前为止所有观察值的平均值
        self.avg = 0.0  # 目前为止所有观察值的总和
        self.cnt = 0    # 目前为止所有观察值的数量

    def update(self, val, n=1):  # val：观察值   n：观察值的数量
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

