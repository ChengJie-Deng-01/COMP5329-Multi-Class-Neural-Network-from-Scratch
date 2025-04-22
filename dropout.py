import numpy as np

class Dropout:
    def __init__(self, rate=0.5, inverted=True):
        """
        初始化 Dropout 层
        :param rate: dropout 率，即神经元被丢弃的概率
        :param inverted: 是否使用 inverted dropout
        """
        self.rate = rate
        self.inverted = inverted
        self.mask = None
        self.scale = 1.0 / (1.0 - rate) if inverted else 1.0
    
    def forward(self, inputs, training=True):
        """
        前向传播
        :param inputs: 输入数据
        :param training: 是否在训练模式
        :return: 经过 dropout 处理后的输出
        """
        if not training:
            return inputs * self.scale if self.inverted else inputs
        
        # 生成随机掩码
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
        
        # 应用 dropout
        output = inputs * self.mask
        
        # 如果是 inverted dropout，需要缩放输出
        if self.inverted:
            output *= self.scale
        
        return output
    
    def backward(self, dvalues):
        """
        反向传播
        :param dvalues: 上一层的梯度
        :return: 经过 dropout 处理后的梯度
        """
        # 应用相同的掩码
        dvalues = dvalues * self.mask
        
        # 如果是 inverted dropout，需要缩放梯度
        if self.inverted:
            dvalues *= self.scale
        
        return dvalues 