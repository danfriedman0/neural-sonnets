# Configuration class for the LSTM model

class Config(object):
  def __init__(self, max_grad_norm, num_layers, hidden_size, max_epochs, max_max_epoch, dropout, batch_size, embed_size,token_type):
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.max_epochs = max_epochs
    self.max_max_epoch = max_max_epoch
    self.dropout = dropout
    self.batch_size = batch_size
    self.embed_size = embed_size
    self.token_type = token_type