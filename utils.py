import torch
import numpy as np


def intersperse(i, xs):
  for x in xs[:-1]:
    yield x
    yield i

  yield xs[-1]


def regress(localnet, globalnet, feats, outsize, lengths):
  tmp = localnet(feats)

  for i in range(lengths):
    tmp[i , 0 , lengths[i]:] = 0.0

  sums = torch.sum(tmp, axis=2)
  outs = globalnet(sums)
  mus = outs[: , :outsize]
  rest = outs[:, outsize:]
  cov = uncholesky(uppertriangle(rest, outsize))
  return (mus, cov)



# good god.
def permutecov(cov, n):
  leftbot = cov[:,0:n,0:n]
  leftmid = cov[:,0:n,n:2*n]
  lefttop = cov[:,0:n,2*n:]

  midbot = cov[:,n:2*n,0:n]
  midmid = cov[:,n:2*n,n:2*n]
  midtop = cov[:,n:2*n,2*n:]

  rightbot = cov[:,2*n:,0:n]
  rightmid = cov[:,2*n:,n:2*n]
  righttop = cov[:,2*n:,2*n:]

  top = torch.cat([midtop, lefttop, righttop], axis=1)
  mid = torch.cat([midbot, leftbot, rightbot], axis=1)
  bot = torch.cat([midmid, leftmid, rightmid], axis=1)

  out = torch.cat([bot, mid, top], axis=2)
  return out



def distloss(targs, mus, cov):

  invcov = torch.linalg.inv(cov)

  deltas = (targs - mus).unsqueeze(dim=1)
  deltasT = torch.transpose(deltas, dim0=1, dim1=2)

  tmp = torch.matmul(invcov, deltasT)

  res = torch.matmul(deltas, tmp).squeeze(2).squeeze(1)

  return res


def loss(targets, mus, cov):
  d = distloss(targets, mus, cov)

  eigs = torch.nn.functional.relu(torch.real(torch.linalg.eigvals(cov)))
  logdet = torch.sum(torch.log(eigs), axis=1)

  return mus , cov , logdet + d
  

def covariance(sigmas, correlations):
  nsig = sigmas.size()[1]
  mul1 = torch.cat([sigmas.unsqueeze(dim=2)]*nsig, axis=2)
  mul2 = torch.transpose(mul1, dim0=1, dim1=2)

  return correlations * mul1 * mul2


def fromuppertriangle(xs, n):
  if n == 1:
    return xs.unsqueeze(dim=2)

  else:
    diag = xs[:,:1].unsqueeze(dim=2)
    offdiag = xs[:,1:n].unsqueeze(dim=1)
    rest = fromuppertriangle(xs[:,n:], n-1)

    row = torch.cat([diag, offdiag], axis=2)
    rect = torch.cat([torch.transpose(offdiag, dim0=1, dim1=2), rest], axis=2)
    ret = torch.cat([row, rect], axis=1)
    return ret


def uppertriangle(xs, n):
  if n == 1:
    return xs.unsqueeze(dim=2)

  else:
    diag = xs[:,:1].unsqueeze(dim=2)
    offdiag = xs[:,1:n].unsqueeze(dim=1)
    rest = uppertriangle(xs[:,n:], n-1)

    row = torch.cat([diag, offdiag], axis=2)
    rect = torch.cat([torch.zeros_like(torch.transpose(offdiag, dim0=1, dim1=2)), rest], axis=2)
    ret = torch.cat([row, rect], axis=1)
    return ret


def uncholesky(m):
  return torch.matmul(m, torch.transpose(m, dim0=1, dim1=2))


def inrange(mn , xs , mx):
  return np.logical_and(mn < xs, xs < mx)



def fitGauss(xs):
  avg = np.mean(xs)
  std = np.std(xs)

  return avg, std



def centralGauss(xs):
  avg = np.mean(xs)
  std = np.std(xs)

  masked = xs[inrange(-2*std , xs , 2*std)]

  return fitGauss(masked)


def binnedGauss(bins, xs, ys):
  means = []
  stds = []
  outbincenters = []
  outbinwidths = []
  for i in range(len(bins)-1):
    mask = inrange(bins[i], xs, bins[i+1])
    if not np.any(mask):
      continue

    # mean , std = fitGauss(ys[mask])
    mean , std = centralGauss(ys[mask])
    means.append(mean)
    stds.append(std)
    outbincenters.append((bins[i] + bins[i+1]) / 2)
    outbinwidths.append((bins[i+1] - bins[i]) / 2)
    continue

  return outbincenters, outbinwidths, means , stds
