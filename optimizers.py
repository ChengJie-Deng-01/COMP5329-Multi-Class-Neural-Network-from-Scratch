import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def update(self, layer):
        raise NotImplementedError("子类必须实现update方法")
    
    def _apply_weight_decay(self, weights):
        if self.weight_decay > 0:
            return weights - self.learning_rate * self.weight_decay * weights
        return weights



class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
    
    def update(self, layer, dweights, dbiases):
        # 应用权重衰减
        layer.weights = self._apply_weight_decay(layer.weights)
        # 更新权重和偏置
        layer.weights -= self.learning_rate * dweights
        layer.bias -= self.learning_rate * dbiases




class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.v_weights = None
        self.v_biases = None
    
    def update(self, layer, dweights, dbiases):
        if self.v_weights is None:
            self.v_weights = np.zeros_like(layer.weights)
            self.v_biases = np.zeros_like(layer.bias)
        
        # 应用权重衰减
        layer.weights = self._apply_weight_decay(layer.weights)
        
        # 更新速度
        self.v_weights = self.momentum * self.v_weights - self.learning_rate * dweights
        self.v_biases = self.momentum * self.v_biases - self.learning_rate * dbiases
        
        # 更新参数
        layer.weights += self.v_weights
        layer.bias += self.v_biases

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0
    
    def update(self, layer, dweights, dbiases):
        if self.m_weights is None:
            self.m_weights = np.zeros_like(layer.weights)
            self.v_weights = np.zeros_like(layer.weights)
            self.m_biases = np.zeros_like(layer.bias)
            self.v_biases = np.zeros_like(layer.bias)
        
        self.t += 1
        
        # 应用权重衰减
        layer.weights = self._apply_weight_decay(layer.weights)
        
        # 更新一阶矩估计
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dweights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * dbiases
        
        # 更新二阶矩估计
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dweights ** 2)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (dbiases ** 2)
        
        # 计算偏差修正
        m_weights_hat = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_hat = self.v_weights / (1 - self.beta2 ** self.t)
        m_biases_hat = self.m_biases / (1 - self.beta1 ** self.t)
        v_biases_hat = self.v_biases / (1 - self.beta2 ** self.t)
        
        # 更新参数
        layer.weights -= self.learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
        layer.bias -= self.learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.epsilon) 