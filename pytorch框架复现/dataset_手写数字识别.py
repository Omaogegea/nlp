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

class Linear:
    def __init__(self,in_feature,out_feature):
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.weight=np.random.normal(0,1,size=(in_feature,out_feature))
        self.bias = np.zeros((1,out_feature))

    def forward(self,x):
        pre=x@self.weight+self.bias
        self.x=x
        return pre

    def backward(self,G):
        dalta_w=self.x.T@G
        dalta_b=np.sum(G,axis=0,keepdims=True)

        self.weight=self.weight-lr*dalta_w
        self.bias = self.bias - lr * dalta_b

        dalta_x=G@self.weight.T
        return dalta_x

class Sigmoid:
    def __init__(self):
        pass
    def forward(self,x):
        self.result=sigmoid(x)
        return self.result

    def backward(self,G):
        result=G*(self.result*(1-self.result))
        return result

class Softmax:
    def __init__(self):
        pass

    def forward(self,x):
        self.p=softmax(x)
        return self.p

    def backward(self,lable):
        return (self.p-lable)/len(lable)




if __name__ == "__main__":
    train_images = load_images(os.path.join("data","train-images.idx3-ubyte"))/255
    train_labels = make_onehot(load_labels(os.path.join("data","train-labels.idx1-ubyte")))

    dev_images = load_images(os.path.join("data","t10k-images.idx3-ubyte"))/255
    dev_labels = load_labels(os.path.join("data","t10k-labels.idx1-ubyte"))

    dev_images = dev_images.reshape(-1, 784)
    train_images = train_images.reshape(60000, 784)

    # w1 = np.random.normal(0,1,size=(784,256))
    # w2 = np.random.normal(0, 1, size=(256, 128))
    # w3 = np.random.normal(0,1,size=(128,10))
    # b1 =np.random.normal(0,1,size=(1,256))
    # b2 = np.random.normal(0, 1, size=(1, 128))
    # b3 = np.random.normal(0, 1, size=(1, 10))
    linear1=Linear(784,256)
    sigmoid_layer = Sigmoid()
    linear2 = Linear(256, 128)
    linear3 = Linear(128, 10)
    softmax_layer=Softmax()

    epoch = 100
    lr = 0.0001
    batch_size = 1

    dataset=Dataset(train_images,train_labels)
    dataloader=DataLoader(dataset,batch_size,shuffle=False)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    for e in range(epoch):
        for batch_data,batch_lable in dataloader:
            # hidden=batch_data @ w1+b1
            hidden=linear1.forward(batch_data)

            hidden_=sigmoid_layer.forward(hidden)
            # hidden_2=hidden_@w2+b2
            hidden_2=linear2.forward(hidden_)
            # pre= hidden_2@w3+b3
            pre=linear3.forward(hidden_2)

            p=softmax_layer.forward(pre)

            loss=-np.mean(batch_lable*np.log(p))

            G3=(p-batch_lable)/batch_data.shape[0]

            G2=linear3.backward(G3)       #第一层
            # dalta_w3=hidden_2.T@G3
            # G2=G3@hidden_2.T
            dalta_hidden_=linear2.backward(G2)     #第二层
            # dalta_w2=hidden_.T@G2
            # dalta_hidden_=G2@w2.T
            G1=sigmoid_layer.backward(dalta_hidden_)
            # G1=dalta_hidden_*(hidden_*(1-hidden_))    #激活函数sigmoid

            linear1.backward(G1)           #第三层
            # dalta_w1=batch_data.T@G1

            # dalta_hidden_=G4@w3.T
            # dalta_hidden=dalta_hidden_*(hidden_*(1-hidden_))
            # dalta_w2=hidden_.T @ G
            # dalta_w1 = batch_data.T @ dalta_hidden
            # delta_b1=np.sum(dalta_hidden,axis=0,keepdims=True)
            # delta_b2 = np.sum(G, axis=0, keepdims=True)
            # w1=w1-lr*dalta_w1
            # w2 = w2 - lr * dalta_w2
            # b1= b1 - lr * delta_b1
            # b2= b2 - lr * delta_b2
            # print(loss)
        right_num=0
        for batch_data,batch_lable in dev_dataloader:
            hidden=linear1.forward(batch_data)
            hidden_=sigmoid_layer.forward(hidden)
            hidden_2=linear2.forward(hidden_)
            pre=linear3.forward(hidden_2)
            # hidden=batch_data@w1+b1
            # hidden_=sigmoid(hidden)
            # pre=hidden_@w2+b2
            pre_idx=np.argmax(pre,axis=-1)
            right_num+=np.sum(pre_idx==batch_lable)
        acc=right_num/len(dev_dataset)
        print(f'acc:{acc:.3f}')


