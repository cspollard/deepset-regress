import torch
import numpy as np
from VarLenSeq import VarLenSeq


# inserts the element i between each element of xs
# assumes xs contains at least one element
def intersperse(i, xs):
  for x in xs[:-1]:
    yield x
    yield i

  yield xs[-1]


# regress given a local network and a global network and a set of features.
# the outsize is the dimensionality of the means and covariances.
# (we assume globalnet has the correct number of outputs to accommodate this!)
def regress(localnet, globalnet, feats):
  # run the local networks in parallel
  tmp = localnet(feats.tensor)

  # average the outputs of the local networks
  avgs = VarLenSeq( tmp , feats.lengths ).mean1()

  globalinputs = torch.cat([ avgs , torch.log((feats.lengths.unsqueeze(1) + 1.0) / 100.0) ], axis=1)

  # extract the mean and covariance of the regressed posterior
  outs = globalnet(globalinputs)
  mus = torch.exp(outs[: , :1])
  sigmas = outs[: , 1:].unsqueeze(1)

  return (mus, sigmas)


# returns the "distance" component of the gaussian loss function
def distloss(targs, mus, cov):
  invcov = torch.linalg.inv(cov)

  deltas = (targs - mus).unsqueeze(dim=1)
  deltasT = torch.transpose(deltas, dim0=1, dim1=2)

  tmp = torch.matmul(invcov, deltasT)

  res = torch.matmul(deltas, tmp).squeeze(2).squeeze(1)

  return res


def loss(targets, mus, cov):
  # the "distance" component of the gaussian loss
  d = distloss(targets, mus, cov)

  eigs = torch.nn.functional.relu(torch.real(torch.linalg.eigvals(cov)))
  # the log of the deterinant of the covariance is the remaining component of
  # the gaussian loss.
  logdet = torch.sum(torch.log(eigs), axis=1)

  # we need to keep the means, covariances, and the actual loss.
  return logdet + d
  

# convert a 1D array of matrix elements into an upper triangular matrix
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


# I think this is the inverse of uppertriangle
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


# invert the cholesky decomposition of a real matrix
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
