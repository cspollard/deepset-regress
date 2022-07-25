import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
from time import time
from sys import argv
import shutil
import json
# import plotutils
import utils
import numpy


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
patience = config["patience"]
lrfactor = config["lrfactor"]

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
n_bkgs = config["n_bkgs"]
max_size = config["max_size"]
mask_value = config["mask_value"]


from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
import os
runname = os.path.join(outdir, time_suffix)

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)

targlen = 1

rng = np.random.default_rng()

def generate_data(mus, sigs, norms, max_size):
  batches = norms.size
  ns = rng.poisson(norms)

  mus = np.broadcast_to(mus, (max_size, 1, batches)).T
  sigs = np.broadcast_to(sigs, (max_size, 1, batches)).T

  outs = mus + sigs * rng.standard_normal(size=(batches, 1, max_size))

  for i in range(outs.shape[0]):
    outs[i,ns[i]:] = mask_value
  
  return outs


def avg(l):
  s = sum(l)
  return s / len(l)

ntests = 1000

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
  , size=ntests
  )

testsigsigmas = \
  rng.uniform \
  ( low=sig_sigma_range[0]
  , high=sig_sigma_range[1]
  , size=ntests
  )

testsiginputs = generate_data(testsigmus, testsigsigmas, testtargs[:,0], -1)

testbkgnorms = \
  rng.uniform \
  ( low=bkg_norm_range[0]
  , high=bkg_norm_range[1]
  , size=ntests
  )

testbkgmus = \
  rng.uniform \
  ( low=bkg_mu_range[0]
  , high=bkg_mu_range[1]
  , size=ntests
  )

testbkgsigmas = \
  rng.uniform \
  ( low=bkg_sigma_range[0]
  , high=bkg_sigma_range[1]
  , size=ntests
  )

testbkginputs = generate_data(testbkgmus, testbkgsigmas, testbkgnorms, -1)
for i in (range(n_bkgs-1)):
  testbkgnorms = \
    rng.uniform \
    ( low=bkg_norm_range[0]
    , high=bkg_norm_range[1]
    , size=ntests
    )

  testbkgmus = \
    rng.uniform \
    ( low=bkg_mu_range[0]
    , high=bkg_mu_range[1]
    , size=ntests
    )

  testbkgsigmas = \
    rng.uniform \
    ( low=bkg_sigma_range[0]
    , high=bkg_sigma_range[1]
    , size=ntests
    )

  testbkginputs = testbkginputs.cat(generate_data(testbkgmus, testbkgsigmas, testbkgnorms, -1))


testinputs = testsiginputs.cat(testbkginputs)

localnodes = [ 1 ] + localnodes

globalnodes = \
    [ localnodes[-1] + 1 ] \
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
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=lrfactor, patience=patience)

for net in nets:
  net.to(device)

os.mkdir(runname + ".plots")

sumloss = 0
sumdist = 0
for epoch in range(number_epochs):
  
  torch.save(localnet.state_dict(), runname + "/localnet.pth")
  torch.save(globalnet.state_dict(), runname + "/globalnet.pth")

  torch.save(localnet.state_dict(), runname + "/localnet.%03d.pth" % epoch)
  torch.save(globalnet.state_dict(), runname + "/globalnet.%03d.pth" % epoch)

  for net in nets:
    net.training = False

  localnet.zero_grad()
  globalnet.zero_grad()


  if epoch > 0:

    writer.add_scalar("learningrate", optim.param_groups[0]['lr'], global_step=epoch)
    writer.add_scalar("avgloss", sumloss / epoch_size, global_step=epoch)
    writer.add_scalar("avgdist", sumdist / epoch_size, global_step=epoch)
    sched.step(sumloss / epoch_size)


  # mus , cov = utils.regress(localnet, globalnet, testinputs, 1)
  # mus = mus.detach()
  # cov = cov.detach()

  # labels = [ "signalrate" ]
  # binranges = [ sig_norm_range ]

  # plotutils.valid_plots \
  #   ( mus.numpy()
  #   , cov.numpy()
  #   , testtargs
  #   , labels
  #   , binranges
  #   , writer
  #   , epoch
  #   , None
  #   )

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
      ( low=sig_norm_range[0]
      , high=sig_norm_range[1]
      , size=(batch_size, 1)
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

    siginputs = generate_data(sigmus, sigsigmas, targs[:,0], max_size)

    bkgnorms = \
      rng.uniform \
      ( low=bkg_norm_range[0]
      , high=bkg_norm_range[1]
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

    bkginputs = generate_data(bkgmus, bkgsigmas, bkgnorms, max_size)

    for i in (range(n_bkgs-1)):
      bkgnorms = \
        rng.uniform \
        ( low=bkg_norm_range[0]
        , high=bkg_norm_range[1]
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

      bkginputs = bkginputs.cat(generate_data(bkgmus, bkgsigmas, bkgnorms, max_size))


    inputs = siginputs.cat(bkginputs)

    mus , cov = utils.regress(localnet, globalnet, inputs, 1)

    targs = torch.Tensor(targs).detach()

    l = utils.loss(targs, mus, cov)

    loss = l.mean()

    loss.backward()

    if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(allparams, grad_clip)

    optim.step()

    sumloss += loss.detach().item()
    sumdist += torch.sqrt((mus[:,0] - targs[:,0])**2).mean().detach().item()

