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

sig_mu = avg(sig_mu_range) * np.ones(100)
sig_sigma = avg(sig_sigma_range) * np.ones(100)

bkg_mu = avg(bkg_mu_range) * np.ones(100)
bkg_sigma = avg(bkg_sigma_range) * np.ones(100)

test_sig50 = torch.Tensor(generate_data(sig_mu, sig_sigma, np.array([50.0]*100), max_range))
test_sig25 = torch.Tensor(generate_data(sig_mu, sig_sigma, np.array([25.0]*100), max_range))
test_sig05 = torch.Tensor(generate_data(sig_mu, sig_sigma, np.array([05.0]*100), max_range))
test_bkg = torch.Tensor(generate_data(bkg_mu, bkg_sigma, np.array([50.0]*100), max_range))

inputs50 = \
  torch.cat \
  ( [ torch.Tensor(test_sig50) , torch.Tensor(test_bkg) ]
  , axis = 2
  )

inputs25 = \
  torch.cat \
  ( [ torch.Tensor(test_sig25) , torch.Tensor(test_bkg) ]
  , axis = 2
  )

inputs05 = \
  torch.cat \
  ( [ torch.Tensor(test_sig05) , torch.Tensor(test_bkg) ]
  , axis = 2
  )

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

  print("plotting")

  mus , cov = utils.regress(localnet, globalnet, inputs50, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu50", mus[:,0].mean(), global_step=epoch)
  writer.add_scalar("avgcorr50", corr.mean(), global_step=epoch)
  writer.add_scalar("avgsig50", torch.sqrt(cov[:,0,0]).mean(), global_step=epoch)
  writer.add_scalar("spread50", (mus[:,0] - 50).std(), global_step=epoch)

  mus , cov = utils.regress(localnet, globalnet, inputs25, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu25", mus[:,0].mean(), global_step=epoch)
  writer.add_scalar("avgcorr25", corr.mean(), global_step=epoch)
  writer.add_scalar("avgsig25", torch.sqrt(cov[:,0,0]).mean(), global_step=epoch)
  writer.add_scalar("spread25", (mus[:,0] - 25).std(), global_step=epoch)

  mus , cov = utils.regress(localnet, globalnet, inputs05, 2)
  corr = cov[:,0,1] / torch.sqrt(cov[:,0,0] * cov[:,1,1])

  writer.add_scalar("avgmu05", mus[:,0].mean(), global_step=epoch)
  writer.add_scalar("avgcorr05", corr.mean(), global_step=epoch)
  writer.add_scalar("avgsig05", torch.sqrt(cov[:,0,0]).mean(), global_step=epoch)
  writer.add_scalar("spread05", (mus[:,0] - 5).std(), global_step=epoch)

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
      ( [ torch.Tensor(siginputs) , torch.Tensor(bkginputs) ]
      , axis = 2
      )

    inputs.requires_grad = True

    mus , cov = utils.regress(localnet, globalnet, inputs, 2)

    targs = torch.Tensor(targs)
    targs.requires_grad = True

    guesses , _ , l = utils.loss(targs, mus, cov)

    loss = l.mean()

    loss.backward()

    if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(allparams, grad_clip)

    sumloss += loss.item()
    sumdist += torch.sqrt((guesses[:,0] - targs[:,0])**2).mean()

    optim.step()

