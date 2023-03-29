#x**2=3

epoch=1000
x=5
lr=0.001
for e in range(epoch):
    pre=x**2
    lable=3
    loss=(pre-lable)**2
    delta_x=2*(pre-lable)*2*x
    x=x-delta_x*lr
    print(x,loss)