#1,加载数据
#2、Dataste类：1,存储数据（all_text,all_lable）;告诉它去哪里取数据（__iter__）
#3、Data;oader:分发数据（__next__）
#
#
import os
import random

import numpy as np
def read_data(file):
    with open(file,encoding='utf-8') as f:
        all_data=f.read().split("\n")

    all_text=[]
    all_lable=[]
    for i in all_data:
        data=i.split("\t")
        if len(data)==2:
            text,lable=data
            all_text.append(text)
            all_lable.append(lable)
    assert len(all_text)==len(all_lable),"数据与标签长度不一样"
    return all_text,all_lable
class Dataset():
    def __init__(self,all_text,all_lable,word_2_idx,lable_2_idx):
        self.all_text=all_text
        self.all_lable=all_lable
        self.word_2_idx=word_2_idx
        self.lable_2_idx=lable_2_idx

    # def __iter__(self):
    #     dataset=Dataloader(self)
    #     return dataset

    def __getitem__(self, index):
        text = self.all_text[index]
        lable = self.all_lable[index]
        text_2_idx=[self.word_2_idx[w] for w in text]
        lable_2_idx=[self.lable_2_idx[lable]]
        return text_2_idx,lable_2_idx

        # return text,lable


class Dataloader():
    def __init__(self,dataset,batch_size,shuffle):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.course=0
        self.shuffle_idx=[i for i in range(len(self.dataset.all_text))]
        random.shuffle(self.shuffle_idx)
    def __iter__(self):
        return self

    def __next__(self):
        if self.course>=len(self.dataset.all_text):
            raise StopIteration
        if self.shuffle==True:
            batch_idx=self.shuffle_idx[self.course:self.course+self.batch_size]
            batch_data=[self.dataset[i] for i in batch_idx]
        else:
            batch_data=[self.dataset[i] for i in range(self.course,min(self.course+self.batch_size,len(all_text)))]
        batch_text_idx, batch_lable_idx=zip(*batch_data)
        # batch_text= [self.dataset.all_text[i] for i in range(self.course,min(self.course+self.dataset.batch_size,len(all_text)))]
        # batch_lable = [self.dataset.all_lable[i] for i in range(self.course,min(self.course+self.dataset.batch_size,len(all_text)))]
        # batch_text=self.dataset.all_text[self.course:self.dataset.batch_size+self.course]
        # batch_lable = self.dataset.all_lable[self.course: self.dataset.batch_size + self.course]
        self.course += len(batch_text_idx)
        return np.array(batch_text_idx),np.array(batch_lable_idx)

def word_2_index(text):
    word_2_idx={}
    for sentence in text:
        for word in sentence:
            word_2_idx[word]=word_2_idx.get(word,len(word_2_idx))
    return word_2_idx

def lable_2_index(lable):
    lable_idx={}
    for w in lable:
        lable_idx[w]=lable_idx.get(w,len(lable_idx))
    return lable_idx
if __name__ == "__main__":
    all_text,all_lable=read_data(os.path.join("data","train0.txt"))
    word_2_idx,lable_2_idx=word_2_index(all_text),lable_2_index(all_lable)
    batch_size=2
    epoch=10
    dataset=Dataset(all_text,all_lable,word_2_idx,lable_2_idx)
    dataloader=Dataloader(dataset,batch_size,shuffle=True)
    for e in range(epoch):
        for batch_data in dataloader:
            batch_text, batch_lable=batch_data
            print(batch_text,batch_lable)