import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

class PGD():
    def __init__(self, batch_size,num_layers, hidden_size, device, loss_fn=CrossEntropyLoss(), hp={"steps": 10}):
        # super().__init__()#(net, loss_fn)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.loss_fn = loss_fn
        self.steps = hp["steps"]
        self.alpha = hp["alpha"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"

    def reset(self, batch_size):
        self.batch_size = batch_size
        h_t = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return (h_t, h_t), l_t

    def perturb(self, net, X, y, h_t, l_t):
        """ generates adversarial examples to given data points and labels (X, y) based on PGD approach. """

        original_X = X
        
        for i in range(self.steps):
            X.requires_grad_()
            # h_t, l_t = self.reset(X.shape[0])
            # add loop later
            if (i%6==0 or i==self.steps-1):
                h_t, l_t, b_t, outputs, p = net(X, l_t, h_t, last=True)

                _loss = self.loss_fn(outputs, y)
                _loss.backward(retain_graph=True)

                X = X + self.alpha * X.grad.sign()
                diff = torch.clamp(X - original_X, min=-self.epsilon, max=self.epsilon)  # gradient projection
                X = torch.clamp(original_X + diff, min=0.0, max=1.0).detach_()  # to stay in image range [0,1]

            else:
                h_t, l_t, b_t, p = net(X, l_t, h_t, last=False)

        return X
