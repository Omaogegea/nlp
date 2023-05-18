import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


all_data=pd.read_csv("上海二手房价.csv")

prices=all_data["房价（元/平米）"].values.reshape(-1,1)
scaler_prices=MinMaxScaler()
scaler_prices.fit(prices)
price=scaler_prices.transform(prices)

rooms=all_data["室"].values.reshape(-1,1)
scaler_rooms=MinMaxScaler()
scaler_rooms.fit(rooms)
room=scaler_rooms.transform(rooms)

floors=all_data["楼层"].values.reshape(-1,1)
scaler_floors=MinMaxScaler()
scaler_floors.fit(floors)
floor=scaler_floors.transform(floors)

years=all_data["建成年份"].values.reshape(-1,1)
scaler_years=MinMaxScaler()
scaler_years.fit(years)
year=scaler_years.transform(years)

features=np.stack([room,floor,year],axis=-1).squeeze(1)
k=np.array([1.0,1.0,1.0]).reshape(-1,1)
b=0

epoch=100
lr=0.001

for e in range(epoch):
    pre=features@k+b
    mean_loss=np.mean((pre-price)**2)
    G=(pre-price)/pre.shape[0]
    datle_k=features.T@G
    datle_b=np.sum(G)
    k-=datle_k*lr
    b-=datle_b*lr
    print(mean_loss)

while True:
    input_room=np.array(input("请输入房间数：")).reshape(-1,1)
    input_floor=np.array(input("请输入楼层:")).reshape(-1,1)
    input_year=np.array(input("请输入年份：")).reshape(-1,1)

    scaler_room = scaler_rooms.transform(input_room)
    scaler_floor = scaler_floors.transform(input_floor)
    scaler_year = scaler_years.transform(input_year)

    features_input=np.stack([scaler_room,scaler_floor,scaler_year],axis=-1).squeeze(1)
    pre=features_input@k+b
    pre_=np.mean(scaler_prices.inverse_transform(pre))

    print(pre_)

