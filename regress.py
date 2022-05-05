import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from time import time
from sys import argv
import pickle
import shutil
import json
import plotutils
import utils
import numpy as np


print("torch version:", torch.__version__)

if len(argv) < 2:
  print("please provide a json steering file")
  exit(-1)

fconfig = open(argv[1])
config = json.load(fconfig)
fconfig.close()

outdir = config["outdir"]
device = config["device"]

batch_size = config["batch_size"]
epoch_size = config["epoch_size"]
number_epochs = config["number_epochs"]

grad_clip = config["grad_clip"]

lr = config["lr"]


globalnodes = config["globalnodes"]
localnodes = config["localnodes"]

sig_mu_range = config["sig_mu_range"]
sig_sigma_range = config["sig_sigma_range"]
bkg_mu_range = config["bkg_mu_range"]
bkg_sigma_range = config["bkg_sigma_range"]
sig_norm_range = config["sig_norm_range"]
bkg_norm_range = config["bkg_norm_range"]
max_range = config["max_range"]


from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
import os
runname = os.path.join(outdir, time_suffix)

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)

rng = np.random.default_rng()

def generate_data(mus, sigs, norms, max_size):
  batches = norms.size
  ns = rng.poisson(norms)

  mus = np.broadcast_to(mus, (max_size, 1, batches)).T
  sigs = np.broadcast_to(sigs, (max_size, 1, batches)).T

  outs = mus + sigs * rng.standard_normal(size=(batches, 1, max_size))
  for i in range(batches):
    outs[i , 0 , ns[i]:] = 0.0
  
  return outs

def avg(l):
  s = sum(l)
  return s / len(l)

ntests = 50

testsig_mu = avg(sig_mu_range) * np.ones(ntests)
testsig_sigma = avg(sig_sigma_range) * np.ones(ntests)

testbkg_mu = avg(bkg_mu_range) * np.ones(ntests)
testbkg_sigma = avg(bkg_sigma_range) * np.ones(ntests)

def gen(sig, bkg):
  sigmu = sig[0]
  sigsig = sig[1]
  sigrate = sig[2]

  bkgmu = bkg[0]
  bkgsig = bkg[1]
  bkgrate = bkg[2]

  sig = torch.Tensor(generate_data(sigmu, sigsig, np.array([sigrate]*ntests), max_range)).detach()
  bkg = torch.Tensor(generate_data(bkgmu, bkgsig, np.array([bkgrate]*ntests), max_range)).detach()

  return \
    torch.cat \
    ( [ sig , bkg ]
    , axis = 2
    ).detach()


# we want a 2D gaussian PDF
targlen = 2

localnodes = [ 1 ] + localnodes

globalnodes = \
    localnodes[-1:] \
  + globalnodes \
  + [ targlen + (targlen * (targlen+1) // 2) ]

act = torch.nn.LeakyReLU(0.01, inplace=True)

localnet = \
  [ torch.nn.Conv1d(localnodes[i], localnodes[i+1], 1) for i in range(len(localnodes) - 1) ]
  
localnet = torch.nn.Sequential(*(utils.intersperse(act, localnet)))

globalnet = \
  [ torch.nn.Linear(globalnodes[i], globalnodes[i+1]) for i in range(len(globalnodes) - 1) ]
  
globalnet = torch.nn.Sequential(*(utils.intersperse(act, globalnet)))

nets = [localnet , globalnet]

# build the optimisers
allparams = [ p for net in nets for p in net.parameters() ]
optim = torch.optim.Adam(allparams, lr = lr)

for net in nets:
  net.to(device)

os.mkdir(runname + ".plots")

sumloss = 0
sumdist = 0
for epoch in range(number_epochs):

  torch.save(localnet.state_dict(), runname + "/localnet.pth")
  torch.save(globalnet.state_dict(), runname + "/globalnet.pth")

  for net in nets:
    net.training = False

  localnet.zero_grad()
  globalnet.zero_grad()

  print("plotting")

  inputs = gen([testsig_mu, testsig_sigma, 50.0], [testbkg_mu, testbkg_sigma, 50.0])

  mus , cov = utils.regress(localnet, globalnet, inputs, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu50", mus[:,0].mean().item(), global_step=epoch)
  writer.add_scalar("avgcorr50", corr.mean().item(), global_step=epoch)
  writer.add_scalar("avgsig50", torch.sqrt(cov[:,0,0]).mean().item(), global_step=epoch)
  writer.add_scalar("spread50", (mus[:,0] - 50).std().item(), global_step=epoch)

  localnet.zero_grad()
  globalnet.zero_grad()


  inputs = gen([testsig_mu, testsig_sigma, 25.0], [testbkg_mu, testbkg_sigma, 50.0])

  mus , cov = utils.regress(localnet, globalnet, inputs, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu25", mus[:,0].mean().item(), global_step=epoch)
  writer.add_scalar("avgcorr25", corr.mean().item(), global_step=epoch)
  writer.add_scalar("avgsig25", torch.sqrt(cov[:,0,0]).mean().item(), global_step=epoch)
  writer.add_scalar("spread25", (mus[:,0] - 25).std().item(), global_step=epoch)

  localnet.zero_grad()
  globalnet.zero_grad()


  inputs = gen([testsig_mu, testsig_sigma, 5.0], [testbkg_mu, testbkg_sigma, 50.0])

  mus , cov = utils.regress(localnet, globalnet, inputs05, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu05", mus[:,0].mean().item(), global_step=epoch)
  writer.add_scalar("avgcorr05", corr.mean().item(), global_step=epoch)
  writer.add_scalar("avgsig05", torch.sqrt(cov[:,0,0]).mean().item(), global_step=epoch)
  writer.add_scalar("spread05", (mus[:,0] - 5).std().item(), global_step=epoch)

  # insert plotting here.
  if epoch > 0:

    writer.add_scalar("avgloss", sumloss / epoch_size, global_step=epoch)
    writer.add_scalar("avgdist", sumdist / epoch_size, global_step=epoch)

  print("starting epoch %03d" % epoch)

  for net in nets:
    net.training = True

  sumloss = 0
  sumdist = 0
  for batch in range(epoch_size):
    localnet.zero_grad()
    globalnet.zero_grad()

    targs = \
      rng.uniform \
      ( low=(sig_norm_range[0], bkg_norm_range[0])
      , high=(sig_norm_range[1], bkg_norm_range[1])
      , size=(batch_size,2)
      )

    sigmus = \
      rng.uniform \
      ( low=sig_mu_range[0]
      , high=sig_mu_range[1]
      , size=batch_size
      )

    sigsigmas = \
      rng.uniform \
      ( low=sig_sigma_range[0]
      , high=sig_sigma_range[1]
      , size=batch_size
      )

    bkgmus = \
      rng.uniform \
      ( low=bkg_mu_range[0]
      , high=bkg_mu_range[1]
      , size=batch_size
      )

    bkgsigmas = \
      rng.uniform \
      ( low=bkg_sigma_range[0]
      , high=bkg_sigma_range[1]
      , size=batch_size
      )

    siginputs = generate_data(sigmus, sigsigmas, targs[:,0], max_range)
    bkginputs = generate_data(bkgmus, bkgsigmas, targs[:,1], max_range)

    inputs = \
      torch.cat \
      ( [ torch.Tensor(siginputs).detach() , torch.Tensor(bkginputs).detach() ]
      , axis = 2
      )

    mus , cov = utils.regress(localnet, globalnet, inputs, 2)

    targs = torch.Tensor(targs).detach()

    guesses , _ , l = utils.loss(targs, mus, cov)

    loss = l.mean()

    loss.backward()

    if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(allparams, grad_clip)

    sumloss += loss.detach().item()
    sumdist += torch.sqrt((guesses[:,0] - targs[:,0])**2).mean().item()

    optim.step()
