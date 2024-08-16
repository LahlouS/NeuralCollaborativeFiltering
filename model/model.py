import torch
import torch.nn as nn 

class GeneralisedMatrixFactorization(nn.Module):
    def __init__(self, c_len, u_len, embed_size):
        
        super().__init__()
        self.c_len = c_len
        self.u_len = u_len
        self.embed_size = embed_size

        self.c_embed = nn.Embedding(self.c_len, embed_size)
        self.u_embed = nn.Embedding(self.u_len, embed_size)

        # you put identity function and bam you have simple GM
        self.act = nn.Identity()
    
    def forward(self, c_idx, u_idx):

        c_embed = self.c_embed(c_idx)
        u_embed = self.u_embed(u_idx)


        dot = (c_embed * u_embed).sum(dim=1).unsqueeze(dim=1)

        return self.act(dot)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, c_len, u_len, embed_size, layers=[64, 32, 16, 8, 4, 1]):

        super().__init__()
        self.c_len = c_len
        self.u_len = u_len
        self.embed_size = embed_size
        self.layers = layers

        self.c_embed = nn.Embedding(self.c_len, embed_size)
        self.u_embed = nn.Embedding(self.u_len, embed_size)

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.embed_size*2, layers[0]))
        for idx, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
            self.mlp.append(nn.Linear(insize, outsize))
            if idx != len(self.layers) - 1:
                self.mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*self.mlp)    
        
        self.sig = nn.ReLU()

    def forward(self, c_idx, u_idx):
        c_embed = self.c_embed(c_idx)
        u_embed = self.u_embed(u_idx)


        c_u_embed = torch.cat((c_embed, u_embed), dim=1)

        x = self.mlp(c_u_embed)
        x = self.sig(x)
        return x


class NeuMF(nn.Module):
    def __init__(self, c_len, u_len, embed_size, layers=[64, 32, 16, 8, 4, 1], gmf_weights=None, mlp_weights=None):
        super().__init__()
        
        self.c_len = c_len
        self.u_len = u_len
        self.embed_size = embed_size
        self.layers = layers

        self.mlp = MultiLayerPerceptron(self.c_len, self.u_len, self.embed_size, self.layers)
        self.gmf = GeneralisedMatrixFactorization(self.c_len, self.u_len, self.embed_size)
        if gmf_weights is not None:
            print('LOG: loading pre-trained GMF weights')
            state = torch.load(gmf_weights, weights_only=True)
            self.gmf.load_state_dict(state)
        if mlp_weights is not None:
            print('LOG: loading pre-trained MLP weights')
            state = torch.load(mlp_weights, weights_only=True)
            self.mlp.load_state_dict(state)

        self.final_linear = nn.Linear(2, 1)
        self.sig = nn.ReLU()
    
    def forward(self, c_idx, u_idx):
        mlp = self.mlp(c_idx, u_idx)
        gmf = self.gmf(c_idx, u_idx)

        x = torch.cat((mlp, gmf), dim=-1)
        x = self.final_linear(x)
        x = self.sig(x)
        return x

if __name__ == '__main__':
    print('instanciating GMF')
    gmf = GeneralisedMatrixFactorization(20, 40, 32)
    print('gmf test forward: gmf([3, 5, 6], [7, 2, 1]):', gmf([3, 5, 6], [7, 2, 1]))

    print('instanciating MLP')
    mlp = MultiLayerPerceptron(20, 40, 32)
    print('mlp test forward: mlp([1], [2]):', mlp([3, 5, 6], [7, 2, 1]))

    print('instanciating NEUMF')
    nmf = NeuMF(20, 40, 32)

    print('nmf test forward: mlp([1], [2]):', nmf([3, 5, 6], [7, 2, 1]))


    
