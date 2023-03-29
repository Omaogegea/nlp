#(x-5)**2+(x+6)**2=0

epoch=100000
lr=0.0001

x1=8
x2=-10

for e in range(epoch):
    pre=(x1-5)**2+(x2+6)**2
    lable=0
    loss=(pre-lable)**2
    delta_x1=2*(pre-lable)*2*(x1-5)
    delta_x2=2*(pre-lable)*2*(x2+6)
    x1=x1-delta_x1*lr
    x2=x2-delta_x2*lr
    print(x1,x2,loss)

