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


  def sum(self):
    s = torch.zeros(self.tensor.size()[:-1])
    for i in range(self.lengths.size()[0]):
      s[i] = torch.sum(self.tensor[ i , : , :self.lengths[i] ], axis=1)

    return s


  def mean1(self):
    s = self.sum()
    for i in range(self.lengths.size()[0]):
      l = self.lengths[i]
      s[i] = s[i] / (l + 1)

    return s


  def mean(self):
    s = self.sum()
    for i in range(self.lengths.size()[0]):
      l = self.lengths[i]
      if l != 0:
        s[i] = s[i] / l

    return s

