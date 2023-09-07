import py2neo


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
            query = """ match (a:疾病 {名称:"%s"})-[r:疾病的症状]->(b:疾病症状) return b.名称""" % (ents[dis_idx])

            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"症状有：{'、'.join(result_text)}"

        elif  dis_idx != -1 and self.keyword_in(text,self.dis_drug_kw):
            query =  """ match (a:疾病 {名称:"%s"})-[r:疾病使用药品]->(b:药品) return b.名称""" % (ents[dis_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"可以服用以下药物：{'、'.join(result_text)}"


        elif dis_idx != -1 and self.keyword_in(text,self.dis_food_kw):
            query = """ match (a:疾病 {名称:"%s"})-[r:疾病推荐食物]->(b:食物) return b.名称""" % (ents[dis_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"可以食用：{'、'.join(result_text)}"

        elif drug_idx!=-1 and self.keyword_in(text,self.drug_dis_kw):
            query = """ match (a:疾病)-[r:疾病使用药品]->(b:药品  {名称:"%s"}) return a.名称""" % (ents[drug_idx])
            result = clinet.run(query)
            result_text = result.data()
            result_text = [list(i.values())[0] for i in result_text]

            answer = f"该药主要治疗：{'、'.join(result_text)}"

        elif  dis_idx != -1 and self.keyword_in(text,self.dis_cure_kw):
            query =  """ match (a:疾病 {名称:"%s"})-[r:疾病治疗方法]->(b:治疗方法) return b.名称""" % (ents[dis_idx])
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

if __name__ == "__main__":
    clinet = py2neo.Graph("http://127.0.0.1:7474", user="neo4j", password="123456")
    qu = QuestionCls()
    qu.text_2_query("请问布洛芬颗粒可以治疗什么病？",[(2,4,"布洛芬颗粒","药品"),(5,8,"有什么","食物")])