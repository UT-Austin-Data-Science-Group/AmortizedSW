
import torch
import torch.nn as nn
class Linear_Mapping(nn.Module):
    def __init__(self, m,d):
        super(Linear_Mapping, self).__init__()
        self.m = m
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(2*m, 1,bias=True),
        )

    def forward(self,input, detach=True):
        if(detach):
            input = input.detach()
        a=self.f(input.transpose(0, 1)).view(1, self.d)
        a = a / torch.sqrt(torch.sum(a ** 2, dim=1))
        return a.view(1, -1)
class Linear_Mapping_K(nn.Module):
    def __init__(self, m,d,k):
        super(Linear_Mapping_K, self).__init__()
        self.m = m
        self.d = d
        self.k = k
        self.f = nn.Sequential(
            nn.Linear(2*m,  k,bias=True),
        )

    def forward(self, input, detach=True):
        if(detach):
            input = input.detach()
        a=self.f(input.transpose(0, 1))
        a = torch.qr(a)[0]
        return a
class Non_Linear_Mapping(nn.Module):
    def __init__(self, m,d):
        super(Non_Linear_Mapping, self).__init__()
        self.m = m
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(2 * m, 1, bias=True),
        )
        self.h = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d,bias=True),
        )

    def forward(self,input, detach=True):
        if(detach):
            input = input.detach()
        a = self.f(input.transpose(0, 1)).view(1, self.d)
        a = self.h(a.view(1,-1))
        a = a / torch.sqrt(torch.sum(a ** 2, dim=1))
        return a.view(1, -1)

class Non_Linear_Mapping_K(nn.Module):
    def __init__(self, m,d,k):
        super(Non_Linear_Mapping_K, self).__init__()
        self.m = m
        self.d = d
        self.k = k
        self.f = nn.Sequential(
            nn.Linear(2 * m, k, bias=True),
        )
        self.h = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d,bias=True),
        )

    def forward(self,input, detach=True):
        if(detach):
            input = input.detach()
        a = self.f(input.transpose(0, 1))
        a = self.h(a.transpose(0,1)).transpose(0,1)
        a = torch.qr(a)[0]
        return a

class Generalized_Linear_Mapping(nn.Module):
    def __init__(self, m,d):
        super(Generalized_Linear_Mapping, self).__init__()
        self.m = m
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(2*m, 1,bias=True),
        )
        self.g = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d,bias=True),
        )
    def forward(self, input,detach=True):
        if(detach):
            input = input.detach()
        a= self.f(self.g(input).transpose(0, 1)).view(1, self.d)
        a= a/torch.sqrt(torch.sum(a**2, dim=1))
        return a.view(1,-1)


class Generalized_Linear_Mapping_K(nn.Module):
    def __init__(self, m,d,k):
        super(Generalized_Linear_Mapping_K, self).__init__()
        self.m = m
        self.d = d
        self.k = k
        self.f = nn.Sequential(
            nn.Linear(2*m, k,bias=True),
        )
        self.g = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d,bias=True),
        )
    def forward(self, input,detach=True):
        if(detach):
            input = input.detach()
        a= self.f(self.g(input).transpose(0, 1))
        a = torch.qr(a)[0]
        return a


