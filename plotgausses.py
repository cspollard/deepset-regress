import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from time import time
from sys import argv
import shutil
import json
import plotutils
import utils
import numpy as np
import os
from VarLenSeq import VarLenSeq
import pickle


print("torch version:", torch.__version__)

fconfig = open(argv[1])
config = json.load(fconfig)
fconfig.close()

outdir = config["outdir"]
device = config["device"]

batch_size = config["batch_size"]
epoch_size = config["epoch_size"]
number_epochs = config["number_epochs"]
patience = config["patience"]

grad_clip = config["grad_clip"]

lr = config["lr"]

targlen = 1

globalnodes = config["globalnodes"]
localnodes = config["localnodes"]

localnodes = [ 1 ] + localnodes

globalnodes = \
    [ localnodes[-1] + 1 ] \
  + globalnodes \
  + [ targlen + (targlen * (targlen+1) // 2) ]


sig_mu_range = config["sig_mu_range"]
sig_sigma_range = config["sig_sigma_range"]
bkg_mu_range = config["bkg_mu_range"]
bkg_sigma_range = config["bkg_sigma_range"]
sig_norm_range = config["sig_norm_range"]
bkg_norm_range = config["bkg_norm_range"]
n_bkgs = config["n_bkgs"]

runname = argv[2]

act = torch.nn.LeakyReLU(0.01, inplace=True)
localnet = \
  [ torch.nn.Conv1d(localnodes[i], localnodes[i+1], 1) for i in range(len(localnodes) - 1) ]
  
localnet = torch.nn.Sequential(*(utils.intersperse(act, localnet)))

globalnet = \
  [ torch.nn.Linear(globalnodes[i], globalnodes[i+1]) for i in range(len(globalnodes) - 1) ]
  
globalnet = torch.nn.Sequential(*(utils.intersperse(act, globalnet)))


localnet.load_state_dict(torch.load(runname + "/localnet.pth"))
globalnet.load_state_dict(torch.load(runname + "/globalnet.pth"))

localnet.to(device)
globalnet.to(device)

outfolder = argv[3]
try: 
    os.mkdir(outfolder) 
except OSError as error: 
  pass


targlen = 1

rng = np.random.default_rng()

def generate_data(mus, sigs, norms, ntimes=1):
  batches = norms.size
  ns = rng.poisson(norms)
  max_size = np.max(ns)

  mus = np.broadcast_to(mus, (max_size, 1, batches)).T
  sigs = np.broadcast_to(sigs, (max_size, 1, batches)).T

  outs = mus + sigs * rng.standard_normal(size=(batches, 1, max_size))
  ns = torch.tensor(ns, dtype=torch.int)
  

  return VarLenSeq(torch.Tensor(outs).detach(), ns)


def avg(l):
  s = sum(l)
  return s / len(l)

ntests = 25

testsig_mu = avg(sig_mu_range) * np.ones(ntests)
testsig_sigma = avg(sig_sigma_range) * np.ones(ntests)

testbkg_mu = avg(bkg_mu_range) * np.ones(ntests)
testbkg_sigma = avg(bkg_sigma_range) * np.ones(ntests)


testtargs = \
  rng.uniform \
  ( low=sig_norm_range[0]
  , high=sig_norm_range[1]
  , size=(ntests, 1)
  )

testsigmus = \
  rng.uniform \
  ( low=sig_mu_range[0]
  , high=sig_mu_range[1]
  , size=(ntests,1)
  )

testsigsigmas = \
  rng.uniform \
  ( low=sig_sigma_range[0]
  , high=sig_sigma_range[1]
  , size=(ntests,1)
  )

testsiginputs = generate_data(testsigmus[:,0], testsigsigmas[:,0], testtargs[:,0])

testbkgnorms = \
  rng.uniform \
  ( low=bkg_norm_range[0]
  , high=bkg_norm_range[1]
  , size=(ntests, n_bkgs)
  )

testbkgmus = \
  rng.uniform \
  ( low=bkg_mu_range[0]
  , high=bkg_mu_range[1]
  , size=(ntests, n_bkgs)
  )

testbkgsigmas = \
  rng.uniform \
  ( low=bkg_sigma_range[0]
  , high=bkg_sigma_range[1]
  , size=(ntests, n_bkgs)
  )

testbkginputs = generate_data(testbkgmus[:,0], testbkgsigmas[:,0], testbkgnorms[:,0])
for i in (range(n_bkgs-1)):

  testbkginputs = testbkginputs.cat(generate_data(testbkgmus[:,i], testbkgsigmas[:,i], testbkgnorms[:,i]))


testinputs = testsiginputs.cat(testbkginputs)
testmus = np.concatenate([testsigmus, testbkgmus], axis=1)
testsigmas = np.concatenate([testsigsigmas, testbkgsigmas], axis=1)
testnorms = np.concatenate([testtargs, testbkgnorms], axis=1)
mu , cov = utils.regress(localnet, globalnet, testinputs, 1)


for i in range(ntests):
  ins = testinputs.tensor[i][:,:testinputs.lengths[i]]
  fig = \
    plotutils.plotGausses \
    ( np.mgrid[-10:30:41j]
    , ins
    , testmus[i]
    , testsigmas[i]
    , testnorms[i]
    , colors=["blue", "red", "green", "orange", "purple", "cyan", "gray", "brown"]
    , labels=["signal", "bkg1", "bkg2", "bkg3", "bkg4", "bkg5", "bkg6"]
    , xlabel="$x$"
    , ylabel="events / unit"
    , text="$\\lambda^{sig}_{true} = %.1f$; $\\lambda^{sig}_{out} = %.1f \\pm %.1f$"
        % (testtargs[i], mu[i], np.sqrt(cov[i,0].detach()))
    )

  fig.savefig(outfolder + "/distributions%d.pdf" % i)

  with open(outfolder + "/chris_toy_%02d.pkl" % i, "wb") as f:
    pickle.dump(ins.tolist(), f)

  print("toy %02d" % i)
  print("mu =", mu[i][0].item())
  print("std =", torch.sqrt(cov[i][0][0]).item())
  print()



for i in range(5):
  with open("toy_%d.pkl" % i, "rb") as f:
    toy = pickle.load(f)

  toy = torch.Tensor(np.array(toy)).unsqueeze(0).unsqueeze(0)
  lengths = torch.tensor(np.array([toy.size()[2]], dtype=np.int32), dtype=torch.int32)
  toy = VarLenSeq(toy, lengths)

  mu , cov = utils.regress(localnet, globalnet, toy, 1)

  print("toy %d" % i)
  print("mu =", mu[0][0].item())
  print("std =", torch.sqrt(cov[0][0][0]).item())
  print()
