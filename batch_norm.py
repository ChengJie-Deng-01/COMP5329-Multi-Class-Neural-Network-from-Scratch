import numpy as np

class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-5):
        """
        初始化 Batch Normalization 层
        :param momentum: 用于计算移动平均的动量
        :param epsilon: 防止除零的小常数
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None  # 缩放参数
        self.beta = None   # 平移参数
        self.running_mean = None  # 移动平均均值
        self.running_var = None   # 移动平均方差
        self.cache = None  # 缓存用于反向传播
    
    def forward(self, inputs, training=True):
        """
        前向传播
        :param inputs: 输入数据
        :param training: 是否在训练模式
        :return: 经过 Batch Normalization 处理后的输出
        """
        if self.gamma is None:
            # 初始化参数
            self.gamma = np.ones(inputs.shape[1])
            self.beta = np.zeros(inputs.shape[1])
            self.running_mean = np.zeros(inputs.shape[1])
            self.running_var = np.zeros(inputs.shape[1])
        
        if training:
            # 计算当前批次的均值和方差
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # 标准化
            x_hat = (inputs - mean) / np.sqrt(var + self.epsilon)
            
            # 缓存用于反向传播
            self.cache = (inputs, mean, var, x_hat)
        else:
            # 使用移动平均的均值和方差
            x_hat = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # 缩放和平移
        output = self.gamma * x_hat + self.beta
        
        return output
    
    def backward(self, dvalues, learning_rate):
        """
        反向传播
        :param dvalues: 上一层的梯度
        :param learning_rate: 学习率
        :return: 经过 Batch Normalization 处理后的梯度
        """
        inputs, mean, var, x_hat = self.cache
        m = inputs.shape[0]
        
        # 计算梯度
        dgamma = np.sum(dvalues * x_hat, axis=0)
        dbeta = np.sum(dvalues, axis=0)
        
        # 计算 dx_hat
        dx_hat = dvalues * self.gamma
        
        # 计算 dvar
        dvar = np.sum(dx_hat * (inputs - mean) * (-0.5) * (var + self.epsilon) ** (-1.5), axis=0)
        
        # 计算 dmean
        dmean = np.sum(dx_hat * (-1) / np.sqrt(var + self.epsilon), axis=0) + \
                dvar * np.sum(-2 * (inputs - mean), axis=0) / m
        
        # 计算 dinputs
        dinputs = dx_hat / np.sqrt(var + self.epsilon) + \
                 dvar * 2 * (inputs - mean) / m + \
                 dmean / m
        
        # 更新参数
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dinputs 