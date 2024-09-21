import torch

def test_meshgrid():
    x = torch.tensor([1,2,3])
    y = torch.tensor([4,5])
    xx,yy = torch.meshgrid(x,y)
    print("xx\n")
    print(xx)
    print("yy\n")
    print(yy)

def test_view():
    x = torch.range(0,11)
    print(f"x: {x.shape}")

if __name__ == '__main__':
    # test_meshgrid()
    test_view()