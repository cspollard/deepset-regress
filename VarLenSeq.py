import torch
import einops

class VarLenSeq:
  # carries a tensor of shape (batch_size , feat_length , seq_length)
  # and a length vector of shape (batch_size,)
  def __init__(self, ten, lengths):
    self.tensor = ten
    self.lengths = lengths
    return


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
    nbatch , nfeat , seqlen = self.tensor.shape

    # choose a random ordering of indices to effectively shuffle the tensor
    idxs = torch.randperm(seqlen)
    idxs = einops.repeat(idxs, "i -> b i", b=nbatch)

    lens = einops.repeat(self.lengths, "b -> b i", i=seqlen)
    mask = idxs < lens

    # if we're truncating, then we need to throw away even more.
    # and also reweight to account for the truncation.
    if 0 < truncated:
      mask = mask * (idxs < truncated) * lens / truncated

    mask = einops.repeat(mask, "b i -> b f i", f=nfeat)

    return einops.reduce(self.tensor * mask, "b f s -> b f", "sum")
