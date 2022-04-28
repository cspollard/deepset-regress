import torch
from sys import argv
import json
from IO import *
import plotutils
import utils
import os


print("torch version:", torch.__version__)

if len(argv) < 2:
  print("usage:")
  print("$ python plot.py config.json model_path out_path")
  exit(-1)

fconfig = open(argv[1])
config = json.load(fconfig)
fconfig.close()

device = config["device"]

tnames = config["tnames"]

valid_frac = config["valid_frac"]

dset = Dataset(tnames, valid_frac=valid_frac)

targlen = dset.targs.size()[1]

regressnodes = \
    [ 20 ] \
  + config["regressnodes"] \
  + [ targlen + (targlen * (targlen+1) // 2) ]


act = torch.nn.LeakyReLU(0.01, inplace=True)
regression = utils.network(regressnodes, act=act, init=dset.normfeats)

networks = [regression]

for net in networks:
  net.to(device)


runname = argv[2]

regression.load_state_dict(torch.load(runname + "/regression.pth"))

outfolder = argv[3]
try: 
    os.mkdir(outfolder) 
except OSError as error: 
  pass



labels = \
  [ "px1", "py1", "pz1"
  , "px2", "py2", "pz2"
  , "pt1", "pt2"
  , "pttot", "invm"
  ]

binranges = \
  [(-500, 500), (-500, 500), (-500, 500)]*2 + [(0, 2000)]*4


targs = dset.targs
mus , cov = \
  utils.regress \
  ( regression
  , dset.feats
  , targlen
  )

mus , cov , l = utils.loss(targs, mus, cov, dset.permutedtargs)


plotutils.valid_plots \
  ( mus.detach().numpy()
  , cov.detach().numpy()
  , targs.detach().numpy()
  , labels
  , binranges
  , None
  , None
  , outfolder
  )


targs = dset.validtargs
mus , cov = \
  utils.regress \
  ( regression
  , dset.validfeats
  , targlen
  )

mus , cov , l = utils.loss(targs, mus, cov, dset.permutedtargs)

plotutils.valid_plots \
  ( mus.detach().numpy()
  , cov.detach().numpy()
  , targs.detach().numpy()
  , labels
  , binranges
  , None
  , None
  , outfolder
  , prefix="valid_"
  )


mask = dset.targs[:,-1] > 2e2
targs = dset.targs[mask]
feats = dset.feats[mask]

mus , cov = \
  utils.regress \
  ( regression
  , feats
  , targlen
  )

mus , cov , l = utils.loss(targs, mus, cov, dset.permutedtargs)

plotutils.valid_plots \
  ( mus.detach().numpy()
  , cov.detach().numpy()
  , targs.detach().numpy()
  , labels
  , binranges
  , None
  , None
  , outfolder
  , prefix="m_gt_200_"
  )