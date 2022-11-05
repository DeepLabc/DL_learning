import torch
import torch.nn as nn

class Gaussian(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = torch.exp((-0.5*input**2)/0.1**2)
        ctx.save_for_backward(result)
        return result
 
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * (-((result*torch.exp(-(result**2)/2*0.1**2))/0.1**2))

        return grad_input

class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.g = Gaussian()

    def forward(self, x):
        print(x)
        print("--------------------")
        x = self.fc1(x)
        out1 = self.relu(x)
        print(out1)
        print("--------------------")
        out2 = self.g.apply(x)
        print(out2)
        return out2


if __name__ == "__main__":
    dummy = torch.randn(3, 4)
    net = myNet()
    out = net(dummy)

