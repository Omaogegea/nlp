import numpy as np
import os
import struct
import matplotlib.pyplot as plt


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28,28)

def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def make_onehot(x):
    onehot=np.zeros((x.shape[0],10))
    for idx, lable in enumerate(x):
        onehot[idx][lable]=1
    return onehot

class Dataset():    
    def __init__(self,train_data,lable_data):
        self.train_data=train_data
        self.lable_data=lable_data

    def __getitem__(self, idx):
        train_idx=self.train_data[idx]
        lable_idx=self.lable_data[idx]
        return train_idx,lable_idx

    def __len__(self):
        return len(self.lable_data)

class DataLoader():
    def __init__(self,dataset,batch_size,shuffle):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle


    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor>=len(self.dataset):
            raise StopIteration
        batch_data=[]
        batch_lable=[]
        for i in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                break
            data=self.dataset[self.cursor]
            batch_data.append(data[0])
            batch_lable.append((data[1]))
            self.cursor+=1

        return np.array(batch_data),np.array(batch_lable)
class Module:
    def __init__(self):
        self.params=[]

    def __repr__(self):
        return self.info

def softmax(x):
    max_x = np.max(x,axis = -1,keepdims=True)
    x = x - max_x
    ex=np.exp(x)
    ex_num=np.sum(ex,axis=1,keepdims=True)
    result=ex/ex_num
    result=np.clip(result,1e-10,1e3)
    return result

def sigmoid(x):
    x = np.clip(x,-100,1e10)
    return 1/(1+np.exp(-x))

class Linear(Module):
    def __init__(self,in_feature,out_feature):
        super().__init__()
        self.info = f'Linear({in_feature} {out_feature})'
        self.W=Parameters(np.random.normal(0,1,size=(in_feature,out_feature)))
        self.B = Parameters(np.zeros((1,out_feature)))
        self.params.append(self.W)
        self.params.append(self.B)

    def forward(self,x):
        pre=x@self.W.weight+self.B.weight
        self.x=x
        return pre

    def backward(self,G):
        self.W.grad=self.x.T@G
        self.B.grad=np.sum(G,axis=0,keepdims=True)

        # self.W.weight=self.W.weight-lr*self.W.grad
        # self.B.weight = self.B.weight - lr * self.B.grad

        dalta_x=G@self.W.weight.T
        return dalta_x

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.info='Sigmoid()'
    def forward(self,x):
        self.result=sigmoid(x)
        return self.result

    def backward(self,G):
        return G*(self.result*(1-self.result))

class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.info='Softmax()'

    def forward(self,x):
        self.p=softmax(x)
        return self.p

    def backward(self,lable):
        return (self.p-lable)/len(lable)

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

class Parameters():
    def __init__(self,weight):
        self.weight = weight
        self.grad = np.zeros_like(self.weight)


class Dropout(Module):
    def __init__(self, rate=0.3):
        super().__init__()
        self.rate = rate
        self.info = f"Dropout({rate})"

    def forward(self, x):
        # r = np.random.rand(x.shape[0],x.shape[1])
        r = np.random.rand(*x.shape)
        self.negtive = r < self.rate

        x[self.negtive] = 0
        return x

    def backward(self, G):
        G[self.negtive] = 0
        return G

class SGD():
    def __init__(self,parameters,lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.weight -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

class MSGD():
    def __init__(self,parameters,lr):
        self.parameters = parameters
        self.lr = lr
        self.u=0.2
        for param in self.parameters:
            param.last_grad = 0

    def step(self):
        for param in self.parameters:
            param.weight -= self.lr * (self.u*param.grad+(1-self.u)*param.last_grad)
            param.last_grad=param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

class Model:
    def __init__(self):
        self.model_list = ModuleList(
            [
                Linear(784, 256),
                Sigmoid(),
                Dropout(0.2),
                Linear(256, 128),
                Dropout(0.1),
                Linear(128, 10),
                Softmax()
            ])

    def forward(self, x, label=None):
        pre = self.model_list.forward(x)

        if label is not None:
            self.label = label
            loss = - np.mean(label * np.log(pre))
            return loss
        else:
            return np.argmax(pre, axis=-1)

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
    train_images = load_images(os.path.join("data","train-images.idx3-ubyte"))/255
    train_labels = make_onehot(load_labels(os.path.join("data","train-labels.idx1-ubyte")))

    dev_images = load_images(os.path.join("data","t10k-images.idx3-ubyte"))/255
    dev_labels = load_labels(os.path.join("data","t10k-labels.idx1-ubyte"))

    dev_images = dev_images.reshape(-1, 784)
    train_images = train_images.reshape(60000, 784)

    epoch = 100
    lr = 0.001
    batch_size = 10

    model=Model()
    opt = MSGD(model.parameters(), lr=lr)

    dataset=Dataset(train_images,train_labels)
    dataloader=DataLoader(dataset,batch_size,shuffle=False)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    for e in range(epoch):
        for x,batch_lable in dataloader:
            loss=model.forward(x,batch_lable)
            model.backward()

            opt.step()
            opt.zero_grad()

        right_num=0
        for batch_data,batch_lable in dev_dataloader:
            pre_idx=model.forward(batch_data)

            right_num+=np.sum(pre_idx==batch_lable)
        acc=right_num/len(dev_dataset)
        print(f'acc:{acc:.3f}')


