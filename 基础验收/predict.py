import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import torch.optim
import pickle
import cv2

def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28, 28)

def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def make_onehot(labels, class_num):
    result = np.zeros((labels.shape[0], class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result

class Dataset:
    def __init__(self, all_images, all_labels):
        self.all_images = all_images
        self.all_labels = all_labels

    def __getitem__(self, index):
        image = self.all_images[index]
        label = self.all_labels[index]

        return image, label

    def __len__(self):
        return len(self.all_images)

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        batch_imges = []
        batch_labels = []
        for i in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                break

            data = self.dataset[self.cursor]

            batch_imges.append(data[0])
            batch_labels.append(data[1])

            self.cursor += 1
        return np.array(batch_imges), np.array(batch_labels)

def softmax(x):
    max_x = np.max(x,axis = -1,keepdims=True)
    x = x - max_x

    # x = np.clip(x, -1e10, 100)
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)

    result = ex / sum_ex

    result = np.clip(result, 1e-10, 1e10)
    return result

def sigmoid(x):
    x = np.clip(x, -100, 1e10)
    result = 1 / (1 + np.exp(-x))
    return result

class Module:
    def __init__(self):
        self.info = "Module:\n"
        self.params = []

    def __repr__(self):
        return self.info

class Parameters():
    def __init__(self,weight):
        self.weight = weight
        self.grad = np.zeros_like(self.weight)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.info += f"**   Linear({in_features},{out_features})"

        self.W = Parameters(np.random.normal(0, 1, size=(in_features, out_features)))
        self.B = Parameters(np.zeros((1, out_features)))

        self.params.append(self.W)
        self.params.append(self.B)

    def forward(self, x):
        result = x @ self.W.weight + self.B.weight

        self.x = x
        return result

    def backward(self, G):
        self.W.grad += self.x.T @ G
        self.B.grad += np.sum(G, axis=0)

        delta_x = G @ self.W.weight.T

        return delta_x

class Conv2D(Module):
    def __init__(self,in_channel,out_channel):
        super(Conv2D, self).__init__()
        self.info += f"     Conv2D({in_channel, out_channel})"
        self.W = Parameters(np.random.normal(0, 1, size=(in_channel, out_channel)))
        self.B = Parameters(np.zeros((1, out_channel)))

        self.params.append(self.W)
        self.params.append(self.B)


    def forward(self, x):
        result = x @ self.W.weight + self.B.weight

        self.x = x
        return result

    def backward(self, G):
        self.W.grad += self.x.T @ G
        self.B.grad += np.sum(G, axis=0)

        delta_x = G @ self.W.weight.T

        return delta_x

class Conv1D(Module):
    def __init__(self,in_channel,out_channel):
        super(Conv1D, self).__init__()
        self.info += f"     Conv1D({in_channel,out_channel})"
        self.W = Parameters(np.random.normal(0, 1, size=(in_channel, out_channel)))
        self.B = Parameters(np.zeros((1, out_channel)))

        self.params.append(self.W)
        self.params.append(self.B)

    def forward(self, x):
        result = x @ self.W.weight + self.B.weight

        self.x = x
        return result

    def backward(self, G):
        self.W.grad += self.x.T @ G
        self.B.grad += np.sum(G, axis=0)

        delta_x = G @ self.W.weight.T

        return delta_x

class SGD():
    def __init__(self,parameters,lr=0.3):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.weight -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

class MSGD():
    def __init__(self,parameters,lr=0.3,u=0.1):
        self.parameters = parameters
        self.lr = lr
        self.u = u

        for param in self.parameters:
            param.last_grad = 0

    def step(self):
        for param in self.parameters:
            # param.weight = param.weight - self.lr * param.grad
            param.weight =   param.weight - self.lr * ((1-self.u)*param.grad + self.u*param.last_grad)
            param.last_grad = param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

class Adam():
    def __init__(self,params,lr=0.01,beta1=0.9,beta2=0.999,e=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.e = e
        self.t = 0

        for p in self.params:
            p.m = 0
            p.v = 0

    def step(self):
        self.t += 1
        for p in self.params:
            gt = p.grad
            p.m = self.beta1 * p.m + (1-self.beta1)*gt
            p.v = self.beta2 * p.v + (1-self.beta2)*gt*gt
            mt_ = p.m / (1-self.beta1**self.t)
            vt_ = p.v / (1-self.beta2**self.t)

            p.weight = p.weight - self.lr * mt_ / (np.sqrt(vt_) + self.e)

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.info += f"**   Sigmoid()"

    def forward(self, x):
        self.result = sigmoid(x)
        return self.result

    def backward(self, G):
        return G * self.result * (1 - self.result)

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.info += f"**    Tanh()"

    def forward(self,x):
        self.result = 2 * sigmoid(2*x) - 1
        return self.result

    def backward(self,G):
        return G * (1 - self.result ** 2)

class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.info += f"**   Softmax()"

    def forward(self, x):
        self.p = softmax(x)
        return self.p

    def backward(self, G):
        G = (self.p - G) / len(G)

        return G

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.info += f"**   ReLU()"

    def forward(self,x):
        self.negative = x < 0
        x[self.negative] = 0
        return x

    def backward(self,G):
        G[self.negative] = 0
        return G

class Dropout(Module):
    def __init__(self,rate=0.3):
        super().__init__()
        self.rate = rate
        self.info += f"**   Dropout({rate})"

    def forward(self,x):
        # r = np.random.rand(x.shape[0],x.shape[1])
        r = np.random.rand(*x.shape)
        self.negtive = r < self.rate

        x[self.negtive] = 0
        return x

    def backward(self,G):
        G[self.negtive] = 0
        return G

class PReLU(Module):
    pass

class ModuleList:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,G):
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __repr__(self):
        info = ""
        for layer in self.layers:
            info += layer.info
            info += "\n"

        return info

class Model:
    def __init__(self):
        self.model_list = ModuleList(
            [
                Linear(784, 256),
                ReLU(),
                Conv2D(256, 10),
                Tanh(),
                Softmax()
            ])

    def forward(self,x,label=None):
        pre = self.model_list.forward(x)

        if label is not None:
            self.label = label
            loss = - np.mean(label * np.log(pre))
            return loss
        else:
            return np.argmax(pre,axis=-1)

    def backward(self):
        self.model_list.backward(self.label)

    def __repr__(self):
        return self.model_list.__repr__()

    def parameters(self):
        all_parameters = []
        for layer in self.model_list.layers:
            all_parameters.extend(layer.params)

        return all_parameters


if __name__ == "__main__":
    with open("0.9406.pkl","rb") as f:
        model = pickle.load(f)

    root_path = "images2"
    images_file = os.listdir(root_path)

    test_img = np.zeros((len(images_file),784))
    test_label = np.zeros((len(images_file),1))
    for fi,file in enumerate(images_file):
        path = os.path.join(root_path,file)
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (28, 28))

        test_img[fi] = img.reshape(-1)
        test_label[fi] = int(file[-5:-4])
    test_img = (test_img) / 255
    test_dataset = Dataset(test_img, test_label)
    test_dataloader = DataLoader(test_dataset, 1, False)
    right = 0
    for x, batch_labels in test_dataloader:
        pre_idx = model.forward(x)

        right += np.sum(int(pre_idx) == int(batch_labels))
    acc = right / len(test_dataset)

    print(acc)