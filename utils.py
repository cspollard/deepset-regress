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
def regress(localnet, globalnet, feats, truncation):
  # run the local networks in parallel
  tmp = localnet(feats.tensor)

  # average the outputs of the local networks
  sums = VarLenSeq( tmp , feats.lengths ).sum(truncation)

  # extract the mean and covariance of the regressed posterior
  outs = globalnet(sums)
  mus = torch.exp(outs[: , 0])
  logsigmas = outs[: , 1]

  return (mus, logsigmas)


def loss(targets, mus, logsigmas):
  return 0.5 * ((mus - targets) / torch.exp(logsigmas))**2 + logsigmas
  

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
