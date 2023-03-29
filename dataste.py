
class MyDataset:
    def __init__(self,all_text,all_lable,batch_size):
        self.all_text=all_text
        self.all_lable=all_lable
        self.batch_size=batch_size
        self.cursor=0

    def __getitem__(self, index):
        if index<len(self):
            text=self.all_text[index]
            lable=self.all_lable[index]
            return text, lable
        else:
            return None,None

    def __iter__(self): # 迭代器，返回一个具有__next__的对象
        return self

    def __next__(self): #可迭代对象
        # 判读取完没有
        if self.cursor>=len(self.all_text):
            raise StopIteration

        #一个batch,一个batch的读取数据
        batch_text=[]
        batch_lable=[]
        for i in range(self.batch_size):
            text,lable=self[self.cursor]
            if text != None:
                batch_text.append(text)
                batch_lable.append(lable)
        #光标后移
            self.cursor+=1
        return batch_text,batch_lable

    def __len__(self):
        return len(self.all_text)

def data_read():
    all_text=["今天天气正好", "晚上的麻辣烫很难吃", "这件衣服很难看", "早上空腹吃早饭不健康", "晚上早点睡觉很健康"]
    all_lable=[1, 0, 0, 0, 1]
    return all_text,all_lable

if __name__ =="__main__":
    all_text,all_lable=data_read()
    batch_size=3
    epoch=10
    dataset=MyDataset(all_text,all_lable,batch_size)

    for i in range(epoch):
        for batch_text,batch_lable in dataset:
            print(batch_text,batch_lable)
