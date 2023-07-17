import torch
import time
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

def read_data(file_name,num=180000):
    with open(file_name,encoding="utf-8") as f:
        all_data = f.read().split("\n")[:-1][:num]

    all_label = []
    all_text = []

    word_2_index = {}

    for text in all_data:
        text_,label_ = text.split("\t")
        for word in text_:
            word_2_index[word] = word_2_index.get(word,len(word_2_index))

        all_text.append(text_)
        all_label.append(int(label_))
    return all_text,all_label,word_2_index


class BertClassifier(nn.Module):
    def __init__(self,class_num):
        super().__init__() # 抄
        self.bert = BertModel.from_pretrained("bert_base_chinese")

        for name,param in self.bert.named_parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

        self.bert.embeddings.word_embeddings

    def forward(self,batch_text,batch_label=None):
        bert_out = self.bert.forward(batch_text)

        bert_out1, bert_out2 = bert_out[0],bert_out[1]
        pre = self.classifier(bert_out2)

        if batch_label is not None:
            loss = self.loss_fun(pre,batch_label)
            return loss
        else:
            return torch.argmax(pre,dim=-1)



if __name__ == "__main__":
    st = time.time()
    train_text,train_label,word_2_index = read_data("..\\data\\train.txt",num=3000)
    index_2_word = list(word_2_index.keys())
    print(time.time() - st)

    dev_text,dev_label,_ = read_data("..\\data\\dev.txt",num=500)
    text_text,text_label,_ = read_data("..\\data\\test.txt")

    epoch = 20

    train_len = len(train_text)
    dev_len = len(dev_text)
    batch_size = 50
    max_len = 10
    class_num = len(set(train_label))
    lr = 0.0001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")
    model = BertClassifier(class_num).to(device)

    optim = torch.optim.AdamW(model.parameters(),lr)

    for e in range(epoch):
        model.train()
        for b_i in range(train_len // batch_size):
            batch_text = train_text[b_i * batch_size : (b_i+1) * batch_size]
            batch_label = train_label[b_i * batch_size : (b_i+1) * batch_size]

            for idx in range(batch_size):
                batch_text[idx] = batch_text[idx][:max_len] # 截断

                batch_text[idx] = tokenizer.encode(batch_text[idx])
                batch_text[idx] = batch_text[idx] + [0] * (max_len + 2 - len(batch_text[idx]))

                # 填充
            batch_text = torch.tensor(batch_text,device=device)
            batch_label = torch.tensor(batch_label,device=device)
            loss = model.forward(batch_text,batch_label)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if b_i % 30 == 0:
                print(f"Loss:{loss:.2f}")


        right = 0
        model.eval()
        for b_i in range(dev_len // batch_size):
            batch_text = dev_text[b_i * batch_size : (b_i+1) * batch_size]
            batch_label = dev_label[b_i * batch_size : (b_i+1) * batch_size]

            for idx in range(batch_size):
                batch_text[idx] = batch_text[idx][:max_len] # 截断

                batch_text[idx] = tokenizer.encode(batch_text[idx])
                batch_text[idx] = batch_text[idx] + [0] * (max_len + 2 - len(batch_text[idx]))

                # 填充
            batch_text = torch.tensor(batch_text,device=device)
            batch_label = torch.tensor(batch_label,device=device)
            pre = model.forward(batch_text)
            right += int(sum(batch_label == pre))
        print(f"acc={right/dev_len * 100} %")



