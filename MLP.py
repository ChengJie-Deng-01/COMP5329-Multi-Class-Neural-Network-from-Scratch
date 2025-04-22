import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from optimizers import SGD, Momentum, Adam
from dropout import Dropout
from batch_norm import BatchNormalization



train_data = np.load("train_data.npy")
train_label= np.load("train_label.npy")


test_data = np.load("test_data.npy")
test_label= np.load("test_label.npy")



# 创建 OneHotEncoder 对象
encoder = OneHotEncoder()

# 对数组进行 One-Hot 编码

train_label_onehot = encoder.fit_transform(train_label).toarray()
test_label_onehot = encoder.fit_transform(test_label).toarray()




# 1: Define Activation Functions
## 1.1: RELU
def relu(inputs):
    
    return np.maximum(0, inputs)

def relu_derivative(inputs):
    return (inputs > 0).astype(float)



## 1.2: GELU
def gelu(inputs):
    
    return 0.5 * inputs * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * np.power(inputs, 3))))

def gelu_derivative(inputs):
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * np.power(inputs, 3))))
    pdf = np.exp(-0.5 * np.power(inputs, 2)) / np.sqrt(2 * np.pi)
    return cdf + inputs * pdf



## 1.3: SOFTMAX
def softmax(inputs):
    
    # 选出每一行的最大值并组成一个向量
    exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # 避免数值爆炸
    
    
    return exps / np.sum(exps, axis=1, keepdims=True)



# 2: Generating Weight Matrix
## 2.1: Initialization (Xavier) 
def xavier_normal(n_inputs, n_neurons):
    
    boundary = np.sqrt(2 / (n_inputs + n_neurons))
    
    return np.random.normal(0, boundary, size=(n_inputs, n_neurons))



def xavier_uniform(n_inputs, n_neurons): # number of inputs（rows），number of neurons this layer（columns）
    
    boundary = np.sqrt(6 / (n_inputs + n_neurons))
    
    return np.random.uniform(-boundary, boundary, size=(n_inputs, n_neurons))






## 2.2: Initialization (He)
def he_normal(n_inputs, n_neurons):
    
    boundary = np.sqrt(2 / n_inputs)
    
    return np.random.normal(0, boundary, size=(n_inputs, n_neurons))


def he_uniform(n_inputs, n_neurons):
    
    boundary = np.sqrt(6 / n_inputs)
    
    return np.random.uniform(-boundary, boundary, size=(n_inputs, n_neurons))




# 3: Initialization Bias
def bias(n_neurons):
    
    return np.zeros(n_neurons, )



# 4: Normalization
def normalization(inputs): # 接收一个array，把最大值缩减成1，小的值按比例缩小
    
    max_number = np.max(np.absolute(inputs), axis=1, keepdims=True) # 找每一行的最大值
    
    scale = np.where(max_number==0, 1, 1/max_number) # where是一个条件，就是如果最大值=0，我们让max_number=1，不然就是1/max_number
    
    return inputs*scale
    
    


## 5: Classification function
def classify(probability): # 输入值是一个概率，输出具体的分类
    
    all_classify = []
    
    for i in probability:
        max_index = np.argmax(i)
        all_classify.append(max_index)

        
    return all_classify
    

## 6: Categorical Cross-Entropy
def categorical_cross_entropy(predicted, real_onehot):
    
    epsilon = 1e-12  # 防止 log(0)
    
    predicted = np.clip(predicted, epsilon, 1. - epsilon)
    
    log_likelihood = -np.sum(real_onehot * np.log(predicted), axis=1)
    
    return log_likelihood








# 定义一个层
class Layer:
    
    # **kwargs 关键字参数解包，可以写成 **anything_you_like，只要前面有 **
    # **后面的optimizer_params 是是一个字典形式的可变关键字参数，接受任意数量的 key=value 对
    # 1. 在函数定义中，显式参数先接收具有相同名字的参数
    # 2. 其余未匹配的命名参数会被打包进 **kwargs（这里是 optimizer_params）
    # 3. 通过 **optimizer_params 再解包，传给另一个函数或类时，会展开为 key=value 形式
    def __init__(self, n_inputs, n_neurons, optimizer='sgd', dropout_rate=0.0, use_batch_norm=False, **optimizer_params): # 如果创建了类的对象，就会自动运行这个，把特定的属性输入进来，self表示类的对象自己
        
        self.weights = np.random.normal(0, np.sqrt(2 / n_inputs), size=(n_inputs, n_neurons)) # He初始化
        
        self.bias = np.zeros(n_neurons, ) # 初始化偏置项

        self.inputs = None
        self.output = None
        
        # 初始化优化器
        if optimizer == 'sgd':

            # 我在def __init__的**optimizer_params对应位置上输入的n个参数都会被传到这里来
            self.optimizer = SGD(**optimizer_params)
        elif optimizer == 'momentum':
            self.optimizer = Momentum(**optimizer_params)
        elif optimizer == 'adam':
            self.optimizer = Adam(**optimizer_params)
        else:
            raise ValueError(f"不支持的优化器: {optimizer}")
        
        # 初始化 Dropout
        self.dropout = Dropout(rate=dropout_rate) if dropout_rate > 0 else None
        
        # 初始化 Batch Normalization
        self.batch_norm = BatchNormalization() if use_batch_norm else None
    

    
    def layer_forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        
        # 应用 Batch Normalization
        if self.batch_norm is not None:
            self.output = self.batch_norm.forward(self.output, training)
        
        # 应用 Dropout
        if self.dropout is not None:
            self.output = self.dropout.forward(self.output, training)
        
        return self.output
    
    def layer_backward(self, dvalues, learning_rate):
        # 应用 Dropout 的反向传播
        if self.dropout is not None:
            dvalues = self.dropout.backward(dvalues)
        
        # 应用 Batch Normalization 的反向传播
        if self.batch_norm is not None:
            dvalues = self.batch_norm.backward(dvalues, learning_rate)
        
        # 计算权重梯度
        dweights = np.dot(self.inputs.T, dvalues)
        # 计算偏置梯度
        dbiases = np.sum(dvalues, axis=0)
        # 计算输入梯度（用于前一层）
        dinputs = np.dot(dvalues, self.weights.T)
        
        # 使用优化器更新参数
        self.optimizer.update(self, dweights, dbiases)
        
        return dinputs


# 定义一个网络

class network():
    
    def __init__(self, network_shape, optimizer='sgd', dropout_rates=None, use_batch_norm=False, **optimizer_params): # network_shape需要表示这个网络由几层组成，每一层有几个神经元
        
        # 比如 network_shape = [128, 64, 32, 16, 10] 表示有五层（第一层是输入），每一层有多少个神经元
        
        self.shape = network_shape
        
        self.layer_lists = [] # 每建立一个新的层都要加进来
        
        # 如果没有指定 dropout_rates，则所有层都不使用 dropout
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(network_shape) - 1)
        
        for i in range( len(network_shape) - 1 ): # 因为权重矩阵的层数比神经元的层数少一个
            
            layer = Layer(network_shape[i], network_shape[i+1], 
                         optimizer=optimizer,
                         dropout_rate=dropout_rates[i],
                         use_batch_norm=use_batch_norm,
                         **optimizer_params) # 第一次定义的是输入层和第一个隐藏层的权重矩阵
            
            self.layer_lists.append(layer)
            
            
    # 前馈运算函数
    
    def forward(self, inputs, acti_func, training=True): # 输出就是不管有多少层都运行完的，直接用Layer类里面的forward函数就可以了
        
        # 定义一个数据的变量，因为每一层的output都不一样
        outputs = [inputs]
        
        for i in range(len(self.layer_lists)):
            
            # 这样第一个循环就是输入到第一个隐藏层的运算
            # layer_ouput = self.layer_lists[i].layer_forward(outputs[i], 'relu') # 每一层的输入都是上一层的输出
            # outputs.append(layer_ouput)
            # outputs = layer_ouput 不能这样写，因为每一层的输出后面都会用到
            
            layer_sum = self.layer_lists[i].layer_forward(outputs[i], training)
            
            if acti_func[i] == "relu":
                
                # 激活函数输出的值可能很大也可能很小，那么一层一层递进的时候，容易梯度爆炸或者梯度消失，因此我们需要对输出进行标准化
                layer_ouput = relu(layer_sum)
                
                # 输出的值不一定是标准化过的，可能很大可能很小，这样会影响网络的稳定性
                
                layer_ouput = normalization(layer_ouput) # 让所有输出值永远都在[0,1]
                
            elif acti_func[i] == "gelu":
                
                layer_ouput = gelu(layer_sum)
                layer_ouput = normalization(layer_ouput)
                
            elif acti_func[i] == "softmax":
                
                layer_ouput = softmax(layer_sum)
                
            
            outputs.append(layer_ouput)
                
        
        return outputs
    

    
    def backward(self, outputs, y_true, acti_func, learning_rate):
        # 计算输出层的梯度
        if acti_func[-1] == "softmax":
            dvalues = outputs[-1] - y_true
        else:
            raise ValueError("最后一层必须是softmax激活函数")
        
        # 反向传播
        for i in range(len(self.layer_lists)-1, -1, -1):
            # 应用激活函数的导数
            if acti_func[i] == "relu":
                dvalues = dvalues * relu_derivative(self.layer_lists[i].output)
            elif acti_func[i] == "gelu":
                dvalues = dvalues * gelu_derivative(self.layer_lists[i].output)
            
            # 更新参数并获取下一层的梯度
            dvalues = self.layer_lists[i].layer_backward(dvalues, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, acti_func, epochs=100, batch_size=128, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批量训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                # 前向传播（训练模式）
                outputs = self.forward(batch_X, acti_func, training=True)
                
                # 反向传播
                self.backward(outputs, batch_y, acti_func, self.layer_lists[0].optimizer.learning_rate)
            
            # 计算验证集损失（非训练模式）
            val_outputs = self.forward(X_val, acti_func, training=False)
            val_loss = np.mean(categorical_cross_entropy(val_outputs[-1], y_val))
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 打印训练信息
            if epoch % 1 == 0:
                train_outputs = self.forward(X_train, acti_func, training=False)
                train_loss = np.mean(categorical_cross_entropy(train_outputs[-1], y_train))
                train_acc = self.evaluate(X_train, y_train, acti_func)
                val_acc = self.evaluate(X_val, y_val, acti_func)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                print(f"Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}")
                print("---"*15)


    def evaluate(self, X, y, acti_func):
        outputs = self.forward(X, acti_func, training=False)
        predictions = classify(outputs[-1])
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        return accuracy

print(100)

n1 = network([128, 64, 32, 16, 10])

print(n1.forward(train_data, ["relu", "gelu", "gelu", "softmax"])[-1])

for i in n1.forward(train_data, ["relu", "gelu", "gelu", "softmax"]):
    
    print(np.shape(i))




np.random.seed(0)
print(n1.forward(train_data, ["relu", "gelu", "gelu", "softmax"])[-1][:5,:])



print(categorical_cross_entropy(n1.forward(train_data, ["relu", "gelu", "gelu", "softmax"])[-1], train_label_onehot))


# 定义激活函数列表
activation_functions = ["gelu", "gelu", "gelu", "softmax"]
network_shape = [128, 64, 64, 32, 10]
dropout_rates = [0., 0., 0., 0]


# 创建网络实例
model_sgd = network([128, 64, 32, 16, 10], optimizer='sgd', learning_rate=0.01)

# model_adam = network([128, 64, 32, 16, 10], optimizer='adam', learning_rate=0.01, beta1=0.9, beta2=0.999)


# 使用 Adam 优化器，添加 Batch Normalization
model_adam = network(network_shape, 
                    optimizer='adam', 
                    dropout_rates=dropout_rates,
                    use_batch_norm=True,  # 启用 Batch Normalization
                    learning_rate=0.001,
                    beta1=0.9,
                    beta2=0.999,
                    weight_decay=0.0003)


# 训练模型
model_adam.train(
    train_data, train_label_onehot,
    test_data, test_label_onehot,
    activation_functions,
    epochs=200,
    batch_size=128
)

print(model_adam.evaluate(test_data, test_label_onehot, activation_functions))