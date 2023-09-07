from transformers import BertModel,BertTokenizer
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from seqeval.metrics import f1_score
from tqdm import tqdm
import ahocorasick
import random
import ahocorasick
import numpy as np
import py2neo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request
import jieba

app = Flask(__name__)


class ReNer:
    def __init__(self,tag_2_index):
        self.tag_2_entity = {}
        self.tag_2_tree = {}

        root_path = os.path.join("data", "entity")
        files = os.listdir(root_path)
        self.type_2_entites = {}
        for file in files:
            tag = file.strip(".txt")

            with open(os.path.join(root_path, file), encoding="utf-8") as f:
                entity = f.read().split("\n")

                self.tag_2_tree[tag] = ahocorasick.Automaton()
                for ent in entity:
                    self.tag_2_tree[tag].add_word(ent, ent)
                self.tag_2_tree[tag].make_automaton()



    def find_entity(self,text):
        result = []
        result_ = []

        for type,tree in self.tag_2_tree.items():
            r = list(tree.iter(text))
            if r:
                for i in r:
                    s,e = i[0]-len(i[1])+1,i[0]+1
                    if (s,e) not in result_:
                        result.append((s,e ,i[1],type))
                        result_.append((s,e ))
        return result

class Entity_align:
    def __init__(self):
        root_path = os.path.join("data", "entity")
        files = os.listdir(root_path)
        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        for file in files:
            tag = file.strip(".txt")

            with open(os.path.join(root_path, file), encoding="utf-8") as f:
                entity = f.read().split("\n")
            tfidf_model = TfidfVectorizer(analyzer="char")
            embs = tfidf_model.fit_transform(entity).toarray()

            self.tag_2_embs[tag] = embs
            self.tag_2_tfidf_model[tag] = tfidf_model

    def align(self,ent_list):
        new_reuslt = []

        for s,e,ent,cls in ent_list:

            ent_emb = self.tag_2_tfidf_model[cls].transform([ent]).toarray()
            sim_score = cosine_similarity(ent_emb,self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]

            if max_score >= 0.5:
                new_reuslt.append((s,e,ent,cls))
        return new_reuslt

class NerModel(nn.Module):
    def __init__(self,lstm_hidden_num,tag_num):
        super(NerModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        for name,param in self.bert.named_parameters():
            # if pooler not in name:
                param.requires_grad =  False

            # if "7" in name:
            #     param.requires_grad = True

        self.lstm = nn.LSTM(768,lstm_hidden_num,batch_first=True,bidirectional=True)

        self.classifier = nn.Linear(lstm_hidden_num*2,tag_num)
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,batch_text_idx,batch_tag_idx=None):
        bert_out0,bert_out1 = self.bert(batch_text_idx,attention_mask=batch_text_idx>0,return_dict=False)

        lstm_out0,list_out1 = self.lstm(bert_out0)
        pre  = self.classifier(lstm_out0)

        if batch_tag_idx != None:
            loss = self.loss_fun(pre.reshape(-1,pre.shape[-1]),batch_tag_idx.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,-1)

def check_tag(ents):
    new_ents = [ents[0]]
    s,e = ents[0][:2]
    for sidx,eidx,ent,type in ents[1:]:
        if  s>=sidx and e<=eidx:
            new_ents.pop()

        if sidx>=s and eidx<=e:
            pass
        else:
            new_ents.append((sidx,eidx,ent,type))
            s, e = sidx,eidx

    return new_ents

def merge(r1,r2):

    r = list(set(r1 + r2))

    r = sorted(r,key=lambda x:(x[0],x[1]))

    result = check_tag(r)

    return result

def find_entities(text,tag_list):
    result = [] # [ (0,2,食品") ,(4,10,"药品") ]

    b_idx = 0
    e_idx = 0
    type = ""

    is_O = True
    for idx, tag in enumerate(tag_list):
        if tag == "O":

            if is_O == True:
                continue
            else:
                e_idx = idx
                if type != "":
                    result.append((b_idx,idx,text[b_idx:idx],type))

                b_idx = -1
                e_idx = -1
                type = ""
            is_O = True
        else:

            if "B-" in tag:

                if is_O == False:
                    e_idx = idx
                    if type != "" :
                        result.append((b_idx, idx,text[b_idx:idx], type))

                    b_idx = -1
                    e_idx = -1
                    type = ""


                b_idx = idx
                type = tag.strip("B-")




            is_O = False
    if b_idx != -1:
        if type != "":
            result.append((b_idx,idx+1,text[b_idx:idx+1],type))


    return  result



class QuestionCls:
    def __init__(self):
        self.dis_sym_kw = ["症状","现象","会怎样","会怎么样","表征"]
        self.dis_drug_kw = ["吃什么药","服用"]
        self.dis_food_kw = ["吃什么东西","饮食","膳食","伙食","食用","补品","食品","菜谱","食谱"]
        self.drug_dis_kw = ["可以治","治疗啥","主治什么"]

        self.dis_cure_kw = ["该怎么办","如何治疗","怎么治疗","咋治","咋办"]

    def text_2_query(self,text,entity_list):
        global clinet
        # entity_list = [(0,4,"XXX","疾病"),(5,9,"XXX","疾病症状")]

        _,_,ents,clss = zip(*entity_list)

        query = ""

        try :
            dis_idx = clss.index("疾病")
        except:
            dis_idx = -1

        try :
            drug_idx = clss.index("药品")
        except:
            drug_idx = -1

        if dis_idx != -1 and self.keyword_in(text,self.dis_sym_kw):
            query = """ match (a:疾病 {名称:"%s"})-[r:症状]->(b:疾病症状) return b.名称""" % (ents[dis_idx])

            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"症状有：{'、'.join(result_text)}"

        elif  dis_idx != -1 and self.keyword_in(text,self.dis_drug_kw):
            query =  """ match (a:疾病 {名称:"%s"})-[r:常用]->(b:药品) return b.名称""" % (ents[dis_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"可以服用以下药物：{'、'.join(result_text)}"


        elif dis_idx != -1 and self.keyword_in(text,self.dis_food_kw):
            query = """ match (a:疾病 {名称:"%s"})-[r:推荐]->(b:食物) return b.名称""" % (ents[dis_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"可以食用：{'、'.join(result_text)}"

        elif drug_idx!=-1 and self.keyword_in(text,self.drug_dis_kw):
            query = """ match (a:疾病)-[r:常用]->(b:药品  {名称:"%s"}) return a.名称""" % (ents[drug_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"该药主要治疗：{'、'.join(result_text)}"

        elif  dis_idx != -1 and self.keyword_in(text,self.dis_cure_kw):
            query =  """ match (a:疾病 {名称:"%s"})-[r:需要做的检查]->(b:治疗方法) return b.名称""" % (ents[dis_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"可以通过一下方式进行治疗：{'、'.join(result_text)}"

        else:
            answer = "不好意思，没听懂！请重新问。"



        return answer



    def keyword_in(self,text,keyword_lst):
        for word in keyword_lst:
            if word in text:
                return True
        return  False


@app.route("/",methods=["GET","POST"])
def abc():
    global history_text

    if request.method == "GET":
        print("get")
        return render_template("index.html")
    else:
        print("post")
        input_text = request.form.get("inputtext")

        # r = model(inputtext)



        input_text = input_text[:510]
        input_idx = tokenizer.encode(input_text, padding="max_length", max_length=max_len, truncation=True,return_tensors="pt")
        input_idx = input_idx.to(device)
        pre = model.forward(input_idx)
        pre = pre.tolist()[0]
        pre = [index_2_tag[i] for i in pre[1:len(input_text) + 1]]
        model_result = find_entities(input_text, pre)

        re_result = check_tag(rener.find_entity(input_text))

        result = merge(model_result, re_result)

        new_result = ea.align(result) # 请问布洛芬颗粒可以治疗什么病

        answer = qu.text_2_query(input_text, new_result)
        # print(answer)

        history_text += f"我：{input_text}\n" + f"帅哥：{answer}\n\n"

        return render_template("index.html",data=history_text)

if __name__ == "__main__":
    history_text = ""


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.load("best_model.pt",map_location=device)

    tokenizer = BertTokenizer.from_pretrained(os.path.join("..","构建实体识别模型","bert_path"))

    with open(os.path.join("data","index_2_tag.txt"),encoding="utf-8") as f:
        index_2_tag = f.read().split("\n")
        tag_2_index = {k:i for i,k in enumerate(index_2_tag)}

    ea = Entity_align()
    rener = ReNer(tag_2_index)
    max_len = 50

    clinet = py2neo.Graph("http://127.0.0.1:7474", user="neo4j", password="maoyaojun37")
    qu = QuestionCls()

    # while True:
    #     input_text = input("请输入：")
    #     input_text = input_text[:510]
    #     input_idx = tokenizer.encode(input_text, padding="max_length", max_length=max_len, truncation=True,return_tensors="pt")
    #     input_idx = input_idx.to(device)
    #     pre = model.forward(input_idx)
    #     pre = pre.tolist()[0]
    #     pre = [index_2_tag[i] for i in pre[1:len(input_text) + 1]]
    #     model_result = find_entities(input_text, pre)
    #
    #     re_result = check_tag(rener.find_entity(input_text))
    #
    #     result = merge(model_result, re_result)
    #
    #     new_result = ea.align(result) # 请问布洛芬颗粒可以治疗什么病
    #
    #     answer = qu.text_2_query(input_text, new_result)
    #     print(answer)

    app.run(host="127.0.0.1", port=9999, debug=True)