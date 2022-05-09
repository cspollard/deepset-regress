# TODO
# check signal bias vs nsig and nbkg
# check evolution vs bkg and sig overlap
# 

import torch
from sys import argv
import json
import plotutils
import utils
import os
import numpy as np
import matplotlib.figure as figure


print("torch version:", torch.__version__)

if len(argv) < 2:
  print("usage:")
  print("$ python plot.py config.json model_path out_path")
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
max_size = config["max_size"]



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

ntests = 10000

testsig_mu = avg(sig_mu_range) * np.ones(ntests)
testsig_sigma = avg(sig_sigma_range) * np.ones(ntests)

testbkg_mu = avg(bkg_mu_range) * np.ones(ntests)
testbkg_sigma = avg(bkg_sigma_range) * np.ones(ntests)

# we want a 2D gaussian PDF
targlen = 2

localnodes = [ 1 ] + localnodes

globalnodes = \
    localnodes[-1:] \
  + globalnodes \
  + [ targlen + (targlen * (targlen+1) // 2) ]

act = torch.nn.LeakyReLU(0.01, inplace=True)

runname = argv[2]

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



labels = [ "nsignal", "nbkg" ]

binranges = \
  [ (sig_norm_range[0], sig_norm_range[1])
  , (bkg_norm_range[0], bkg_norm_range[1])
  ]

# targs = \
#   rng.uniform \
#   ( low=(sig_norm_range[0], bkg_norm_range[0])
#   , high=(sig_norm_range[1], bkg_norm_range[1])
#   , size=(ntests,2)
#   )

# sigmus = \
#   rng.uniform \
#   ( low=sig_mu_range[0]
#   , high=sig_mu_range[1]
#   , size=ntests
#   )

# sigsigmas = \
#   rng.uniform \
#   ( low=sig_sigma_range[0]
#   , high=sig_sigma_range[1]
#   , size=ntests
#   )

# bkgmus = \
#   rng.uniform \
#   ( low=bkg_mu_range[0]
#   , high=bkg_mu_range[1]
#   , size=ntests
#   )

# bkgsigmas = \
#   rng.uniform \
#   ( low=bkg_sigma_range[0]
#   , high=bkg_sigma_range[1]
#   , size=ntests
#   )


# siginputs = generate_data(sigmus, sigsigmas, targs[:,0], max_size)
# bkginputs = generate_data(bkgmus, bkgsigmas, targs[:,1], max_size)

# inputs = \
#   torch.cat \
#   ( [ torch.Tensor(siginputs).detach() , torch.Tensor(bkginputs).detach() ]
#   , axis = 2
#   )

# mus , cov = \
#   utils.regress \
#   ( localnet
#   , globalnet
#   , inputs
#   , targlen
#   )

# l = utils.loss(torch.Tensor(targs), mus, cov)

# plotutils.valid_plots \
#   ( mus.detach().numpy()
#   , cov.detach().numpy()
#   , targs
#   , labels
#   , binranges
#   , None
#   , None
#   , outfolder
#   )

sigranges = list(range(20, 70))
bkgranges = list(range(10, 80, 10))
bkgranges.reverse()
nreplicas = 100

fig1 = figure.Figure()
ax1 = fig1.add_subplot(111)
fig2 = figure.Figure()
ax2 = fig2.add_subplot(111)
colors = ["red", "blue", "green", "black", "orange", "purple", "cyan", "gray", "brown"]

icolor = 0
for nbkg in bkgranges:
  uncertratios = []
  correlations = []
  for nsig in sigranges:
    siginputs = rng.normal(avg(sig_mu_range), avg(sig_sigma_range), max_size*nreplicas)
    siginputs = siginputs.reshape(nreplicas, 1, max_size)
    siginputs[:,:,nsig:] = 0.0

    bkginputs = rng.normal(avg(bkg_mu_range), avg(bkg_sigma_range), max_size*nreplicas)
    bkginputs = bkginputs.reshape(nreplicas, 1, max_size)
    bkginputs[:,:,nbkg:] = 0.0

    inputs = \
      torch.cat \
      ( [ torch.Tensor(siginputs).detach() , torch.Tensor(bkginputs).detach() ]
      , axis = 2
      )

    mus , cov = \
      utils.regress \
      ( localnet
      , globalnet
      , inputs
      , targlen
      )

    uncertratios.append(torch.sqrt(cov[:,0,0]).mean().item() / np.sqrt(nsig))
    correlations.append((cov[:,1,0] / torch.sqrt(cov[:,0,0] * cov[:,1,1])).mean().item())
    continue

  ax1.plot(sigranges, uncertratios, "-", color=colors[icolor], label="$n_{bkg}^{true} = %d$" % nbkg)
  ax2.plot(sigranges, correlations, "-", color=colors[icolor], label="$n_{bkg}^{true} = %d$" % nbkg)

  icolor = icolor + 1
  continue

ax1.set_xlabel("$n_{sig}^{true}$")
ax1.set_ylabel("$\sigma_{sig}^n / \sqrt{n_{sig}^{true}}$")
ax1.legend()
fig1.savefig(outfolder + "/siguncerts.pdf")

ax2.set_xlabel("$n_{sig}^{true}$")
ax2.set_ylabel("$n_{sig}$ - $n_{bkg}$ correlation")
ax2.legend()

fig2.savefig(outfolder + "/correlations.pdf")
