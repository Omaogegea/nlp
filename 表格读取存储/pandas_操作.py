import pandas as pd


# all_data = pd.read_excel("info.xls")
all_data = pd.read_csv("new_info.csv",sep="#",names=["姓名","年龄","学号"])

all_name = list(all_data["姓名"])
all_num = all_data["学号"].tolist()
all_age = all_data["年龄"].to_list()

all_age = [i+1 for i in all_age]

dataf = pd.DataFrame({"姓名":all_name,"年龄":all_age,"学号":all_num})

# dataf.to_excel("new_info.xlsx")
dataf.to_csv("new_info.csv",sep="#",index=False,header=False)

print("")