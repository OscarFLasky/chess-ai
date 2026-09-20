import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np





n_hidden = 128
n_trans = 162 #18*3*3
n_final = 73
unfold = 3

class Linear(nn.Module):

    def __init__(self, fan_in, fan_out, bias = False, unf=True):
        super().__init__()  
        self.weight = nn.Parameter(torch.randn((fan_out, fan_in))/(fan_in**0.5))
        self.bias = nn.Parameter(torch.zeros((fan_out,1))) if bias else None
        self.unf = unf

    def forward(self, x):
        B = x.shape[0]
        if self.unf:
            C = self.weight.shape[1] // (unfold*unfold)
            w = self.weight.view(-1, C, unfold, unfold)
            return F.conv2d(x, w, padding=1)
        x = x.view(B, x.shape[1], -1)
        out = self.weight @ x
        return out.view(B, -1, 8, 8)

class LinearFlat(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((fan_out, fan_in)) / (fan_in ** 0.5))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
class BatchNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = nn.Parameter(torch.ones(dim, device=device, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(dim, device=device, requires_grad=True))

        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var

        shape = (1, -1, 1, 1)
        xhat = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self.gamma.view(shape) * xhat + self.beta.view(shape)


class Tanh(nn.Module):

    def forward(self,x):
        self.out = torch.tanh(x)
        return self.out


class reLU(nn.Module):

    def forward(self, x):
        out = x * (x>0)
        return out

class ResBlock(nn.Module):
    def __init__(self, n_hidden, k):
        super().__init__()
        self.l1 = Linear(n_hidden * k * k, n_hidden)
        self.bn1 = BatchNorm2d(n_hidden)
        self.conv2 = Linear(n_hidden * k * k, n_hidden)
        self.bn2 = BatchNorm2d(n_hidden)
        self.relu = reLU()

    def forward(self, x):
        out = self.bn1(self.l1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

class ChessNet(nn.Module):

    def __init__(self, n_blocks = 6, n_hidden = 128, n_in = 18, n_final = 73, unfold = 3,
                 value_head = "tanh"):
        super().__init__()
        # "tanh" : sortie entre -1 et 1, apprise en MSE (reseaux d'avant)
        # "logit" : sortie brute, apprise en BCEWithLogits ; valeur = 2*sigmoid(z)-1
        self.value_head = value_head
        self.linIn = Linear(n_in*unfold * unfold , n_hidden)
        self.bnIn = BatchNorm2d(n_hidden)

        self.blocks = nn.ModuleList([ResBlock(n_hidden, unfold) for i in range(n_blocks)])

        self.policyLinOut = Linear(n_hidden, n_final, unf=False)

        self.valueLinOut = Linear(n_hidden, 1, unf=False)
        self.value_bnOut = BatchNorm2d(1)
        self.value_fc1Out = LinearFlat(64, 256, bias=True)
        self.value_fc2Out = LinearFlat(256, 1, bias=True)
        self.relu = reLU()
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Linear) and m is not self.policyLinOut:
                    m.weight *= 2 ** 0.5
            self.policyLinOut.weight *= 0.1

    def forward(self, x):
        B = x.shape[0]
        
        x = self.relu(self.bnIn(self.linIn(x)))

        for block in self.blocks:
            x = block(x)

        logits = self.policyLinOut(x)
        logits = logits.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

        v = self.relu(self.value_bnOut(self.valueLinOut(x)))
        v = v.reshape(B, -1)
        v = self.relu(self.value_fc1Out(v))
        v = self.value_fc2Out(v)
        if self.value_head == "tanh":
            v = torch.tanh(v)

        return logits, v