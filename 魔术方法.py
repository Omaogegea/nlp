class Apple:
    def __init__(self):
        print("实例化的时候直接触发")
        
    def __repr__(self):
        return ("用print的时候触发")

    def __len__(self):  #使用len()的时候触发
        return 100

    def __call__(self, *args, **kwargs):  #*args:未命名变量；**kwargs：命名变量
        return ("用()的时候触发")

    def __getitem__(self, item): #在使用下标的时候会触发
        return 99

if __name__ == "__main__":
    a=Apple()
    print(a)
    b=len(a)
    print(b)
    print(a())
    print(a[0])





        