import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
def data_read(dir):
    with open(dir,encoding="utf-8") as f:
        data=f.read().split("\n")
    all_data=[]
    all_lable=[]
    for i in data:
        data_=i.split("\t")
        if len(data_)!=2:
            continue
        text,lable=data_
        all_data.append(text)
        all_lable.append(int(lable))
    return all_data,all_lable

def make_onehot(lable,class_num):
    onehot=np.zeros((len(lable),class_num))
    for idx,cls in enumerate(lable):
        onehot[idx][cls]=1
    return onehot

def word_2_index(all_data):
    word_2_index={"PAD":0}
    for data in all_data:
        for w in data:
            word_2_index[w]=word_2_index.get(w,len(word_2_index))
    index_2_word=list(word_2_index)
    return word_2_index,index_2_word
def make_word_onehot(word_2_index):
    onehot=np.zeros((len(word_2_index),len(word_2_index)))
    for i in range(len(word_2_index)):
        onehot[i][i] = 1
    return onehot

def softmax(x):
    max_x = np.max(x,axis = -1,keepdims=True)
    x = x - max_x

    # x = np.clip(x, -1e10, 100)
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)

    result = ex / sum_ex

    result = np.clip(result, 1e-10, 1e10)
    return result

class MyDataset(Dataset):
    def __init__(self,all_data,all_lable):
        self.all_data=all_data
        self.all_lable=all_lable

    def __getitem__(self, index):
        text=self.all_data[index][:max_len]
        lable=self.all_lable[index]

        text_idx = [word_2_index[i] for i in text]
        text_idx = text_idx + [0] * (max_len - len(text_idx) )
        text_emb = [word_onehot[j] for j in text_idx]
        text_emb = np.array(text_emb)

        return text_emb,lable

    def __len__(self):
        return len(all_lable)

if __name__ == "__main__":
    all_data,all_lable=data_read(os.path.join("data","文本分类","train.txt"))
    all_lable=make_onehot(all_lable,10)
    word_2_index,index_2_word=word_2_index(all_data)
    word_onehot=make_word_onehot(word_2_index)

    lr=0.01
    batch_size=3
    epoch=10
    max_len = 30

    w1 = np.random.normal(size=(len(word_2_index),10))

    train_dataset=MyDataset(all_data,all_lable)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    for e in range(epoch):
        for batch_text_emb,batch_label in train_dataloader:
            batch_text_emb = batch_text_emb.numpy()
            batch_label = batch_label.numpy()
            pre = batch_text_emb @ w1
            pre_mean = np.mean(pre,axis=1)
            p = softmax(pre_mean)

            loss = - np.sum( batch_label * np.log(p) + (1-batch_label) * np.log(1-p) )

            G = (p-batch_label)/len(pre)

            dpre = np.zeros_like(pre)
            for i in range(len(G)):
                for j in range(G.shape[1]):
                    dpre[i][:,j] = G[i][j]

            delta_w1 = batch_text_emb.transpose(0,2,1) @ dpre

            delta_w1 = np.mean(delta_w1,axis=0)
            w1 = w1 - lr * delta_w1

        print(loss)