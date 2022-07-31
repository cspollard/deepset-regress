import torch

class VarLenSeq:
  # carries a tensor of shape (batch_size, x , l) and a mask of shape
  # (batch_size)
  def __init__(self, ten, lengths):
    self.tensor = ten
    self.lengths = lengths


  def cat(self, vls):
    newlengths = vls.lengths + self.lengths
    maxlen = torch.max(newlengths)
    newsize = list(self.tensor.size())
    newsize[2] = int(maxlen)
    newtensor = torch.zeros(newsize)

    for i in range(self.lengths.size()[0]):
      newtensor[ i , : , :self.lengths[i] ] = self.tensor[ i , : , :self.lengths[i] ]
      newtensor[ i , : , self.lengths[i]:newlengths[i] ] = vls.tensor[ i , : , :vls.lengths[i] ]
      continue

    return VarLenSeq(newtensor, newlengths)


  def sum(self, truncated=-1):
    s = torch.zeros(self.tensor.size()[:-1])

    for batch in range(self.lengths.size()[0]):
      l = self.lengths[batch]
      if 0 < truncated and truncated < l:
        shuffled = self.tensor[batch].T[torch.randperm(l)].T

        truncked = torch.sum(shuffled[ : , :truncated ], axis=1)
        untruncked = torch.sum(shuffled[ : , truncated: ], axis=1).detach()

        s[batch] = truncked + untruncked

      else:
        s[batch] = torch.sum(self.tensor[ batch , : , :l ], axis=1)

    return s
