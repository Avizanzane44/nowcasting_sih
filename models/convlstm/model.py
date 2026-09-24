import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class Seq2SeqConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=(3, 3)):
        super(Seq2SeqConvLSTM, self).__init__()
        # Encoder
        self.encoder = ConvLSTMCell(in_channels, hidden_channels, kernel_size, True)
        # Decoder
        self.decoder = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size, True)
        # Final output projection
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x, future_seq_len):
        """
        x: (batch, seq_len, channels, height, width)
        returns: (batch, future_seq_len, channels, height, width)
        """
        b, seq_len, _, h, w = x.size()
        
        # Initialize hidden state
        h_t, c_t = self.encoder.init_hidden(b, (h, w))
        
        # Encoder pass
        for t in range(seq_len):
            h_t, c_t = self.encoder(x[:, t, :, :, :], (h_t, c_t))
            
        # Decoder pass (using last encoded state)
        outputs = []
        decoder_input = h_t  # Feed last hidden state as first input to decoder
        for t in range(future_seq_len):
            h_t, c_t = self.decoder(decoder_input, (h_t, c_t))
            pred = self.out_conv(h_t)
            outputs.append(pred.unsqueeze(1))
            decoder_input = h_t  # Autoregressive passing of hidden state (or could use pred)
            
        return torch.cat(outputs, dim=1)
