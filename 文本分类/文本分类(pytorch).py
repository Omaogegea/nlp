import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
def data_read(dir,num=None):
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
    if num and num > 0:
        return all_data[:num],all_lable[:num]
    elif num and num < 0 :
        return all_data[num:], all_lable[num:]
    else:
        return all_data, all_lable
    return all_data,all_lable
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
        text_emb = np.array(text_emb,dtype=np.float32)

        return text_emb,lable

    def __len__(self):
        return len(all_lable)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w=nn.Linear(len(word_2_index),10)
        self.loss_fun=nn.CrossEntropyLoss()

    def forward(self,x,lable=None):
        pre=self.w(x)
        pre=torch.mean(pre,dim=1)
        if lable is not None:
            loss= self.loss_fun(pre,lable)
            return loss
        else:
            return torch.argmax(pre,dim=-1)

if __name__ == "__main__":
    all_data,all_lable=data_read(os.path.join("data","文本分类","train.txt"),20000)
    test_data, test_lable = data_read(os.path.join("data", "文本分类", "train.txt"),-300)
    word_2_index, index_2_word = word_2_index(all_data)
    word_onehot = make_word_onehot(word_2_index)

    lr = 0.01
    batch_size = 1
    epoch = 100
    max_len = 30

    device="cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = MyDataset(all_data, all_lable)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = MyDataset(test_data, test_lable)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MyModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_text, batch_label in tqdm(train_dataloader):
            batch_text.to(device)
            batch_label.to(device)
            loss=model.forward(batch_text, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

            right_num=0
        for batch_text, batch_lable in train_dataloader:
            batch_text.to(device)
            batch_label.to(device)
            pre= model.forward(batch_text)

            right_num += int(torch.sum(pre==batch_lable))

        acc=right_num/len(test_dataset)
        print(f"epoch:{e},acc;{acc:.3f}")
