import torch.nn as nn

import modules


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, im_size, hidden_size, num_classes, is_lstm, num_layers, is_covnet
    ):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()

        self.std = std
        self.is_lstm = is_lstm
        if is_covnet:
            self.cnet = modules.CNetwork(g, 128, hidden_size) # for 32*32
            # self.cnet = modules.ContextNetwork(1152, hidden_size, im_size) # 1152 for 64*64, 512 for 32*32
            self.sensor = modules.GlimpseCovNet(h_g, h_l, g, k, s, c) # 64*64
        else:
            self.cnet = modules.ContextNetwork(128, hidden_size, im_size) # 1152 for 64*64, 512 for 32*32
            self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        
        if is_lstm:
            # self.contextNN = modules.ContextNetwork(1152, hidden_size, im_size) # 64*64
            self.contextNN = modules.ContextNetwork(128, hidden_size, im_size)
            self.rnn = modules.CoreRNN(hidden_size, hidden_size, num_layers) 
        else:
            self.rnn = modules.CoreNetwork(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        if self.is_lstm:
            g_t = g_t.unsqueeze(0)
            output, h_t = self.rnn(g_t, h_t_prev)
            log_pi, l_t = self.locator(output[-1].squeeze(0))
            b_t = self.baseliner(output[-1].squeeze(0)).squeeze()
        else:
            h_t = self.rnn(g_t, h_t_prev)
            log_pi, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t).squeeze()

        if last:
            if self.is_lstm:
                log_probas = self.classifier(output[-1].squeeze(0))
            else:
                log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi
