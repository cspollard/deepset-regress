import torch

class VarLenSeq:
  # carries a tensor of shape (batch_size, x , l) and a mask of shape
  # (batch_size)
  def __init__(self, ten, lengths, max_size=-1):
    self.tensor = ten
    self.lengths = lengths
    self.max_size = max_size

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

    return VarLenSeq(newtensor, newlengths, self.max_size)


  def sum(self):
    s = torch.zeros(self.tensor.size()[:-1])
    for i in range(self.lengths.size()[0]):
      if self.max_size > 0:
        l = min(self.max_size, self.lengths[i])
      else:
        l = self.lengths[i]
      s[i] = torch.sum(self.tensor[ i , : , :l ], axis=1)

    return s


  def mean1(self):
    s = self.sum()
    for i in range(self.lengths.size()[0]):
      if self.max_size > 0:
        l = min(self.max_size, self.lengths[i])
      else:
        l = self.lengths[i]
      s[i] = s[i] / (l + 1)

    return s

  def truncated(self):
    newshape = list(self.tensor.size())
    if self.max_size > 0:
      newshape[2] = self.max_size

    tmp = torch.zeros(newshape)
    for i in range(self.lengths.size()[0]):
      if self.max_size > 0:
        l = min(self.max_size, self.lengths[i])
      else:
        l = self.lengths[i]
      idxs = torch.randperm(self.lengths[i])
      permuted = self.tensor[i,:,idxs]
      tmp[i,:,:l] = permuted[:,:l]

    return tmp
