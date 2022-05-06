import numpy as np
import matplotlib.figure as figure
import matplotlib.colors as colors
from itertools import cycle
import utils



def gaussian(mu, sigma, normalized=True):
  sigma2 = sigma**2

  def f(x):
    main = np.exp(- (x - mu)**2 / (2*sigma2))
    if normalized:
      return main / (np.sqrt(2*np.pi)*sigma)
    else:
      return main

  return f


def normal(xs, normalized=True):
  mu = np.mean(xs)
  sigma = np.std(xs)

  return gaussian(mu, sigma, normalized=normalized)


def std1(xs, n):
  mu = np.mean(xs)
  sigma = np.std(xs)
  mask = np.logical_and((mu - n*sigma) < xs, xs < (mu + n*sigma))

  return np.std(xs[mask])


def save_fig(fig, name, writer, epoch, outdir):
  if writer:
    writer.add_figure(name, fig, global_step = epoch)
  if outdir:
    fig.savefig(outdir + "/" + name + ".pdf")

  return


def profile(bins, xs, ys):
    bincenters, binwidths, means , stds = \
      utils.binnedGauss(bins, xs, ys)

    fig = figure.Figure()
    ax = fig.add_subplot(111)
    ax.errorbar \
      ( bincenters
      , means
      , xerr = binwidths
      , yerr = stds
      , marker = '.'
      )

    return fig


def valid_plots(mus, cov, targets, labels, binranges, writer, epoch, outdir, prefix=""):

  # first pair everything up
  sigmas = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

  for i in range(len(labels)):
    bmin = binranges[i][0]
    bmax = binranges[i][1]

    fig = figure.Figure()

    targs = targets[: , i]

    ax = fig.add_subplot(111)
    ax.hist \
      ( targs
      , bins=np.mgrid[bmin:bmax:101j]
      , color="blue"
      , density=True
      )

    name = prefix + "true_%s" % labels[i]
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()

    thesemus = mus[:,i]
    thesesigmas = sigmas[:,i]
    diffs = thesemus - targs

    pulls = diffs / thesesigmas

    ax = fig.add_subplot(111)
    ax.hist \
      ( pulls
      , bins=np.mgrid[-5:5:101j]
      , color="blue"
      , density=True
      )

    xs = np.mgrid[-5:5:100j]
    ys = gaussian(0, 1)(xs)

    ax.plot(xs, ys, "--", color="black")

    name = prefix + "pulls_%s" % labels[i]
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()

    ax = fig.add_subplot(111)
    ax.hist \
      ( diffs
      , bins=np.mgrid[-bmax:bmax:101j]
      , color="blue"
      , density=True
      )

    name = prefix + "diffs_%s" % labels[i]
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()


    name = prefix + "diffs_evolution_%s" % labels[i]
    fig = profile(np.mgrid[bmin:bmax:11j], targs, diffs)
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()


    resps = diffs / targs
    ax = fig.add_subplot(111)
    ax.hist \
      ( resps
      , bins=np.mgrid[-5:5:101j]
      , color="blue"
      , density=True
      )

    xs = np.mgrid[-5:5:100j]

    avg, std = utils.centralGauss(resps)
    ys = gaussian(avg, std)(xs)
    ax.plot(xs, ys, "--", color="black", label="$\\mu = %.2f, \\sigma = %.2f$" % (avg , std) )

    ax.set_xlim(-3, 3)
    ax.legend()

    name = prefix + "response_%s" % labels[i]
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()

    name = prefix + "response_evolution_%s" % labels[i]
    fig = profile(np.mgrid[bmin:bmax:11j], targs, resps)
    save_fig(fig, name, writer, epoch, outdir)
    fig.clf()


    ax = fig.add_subplot(111)
    ax.hist \
      ( thesemus
      , bins=np.mgrid[bmin:bmax:51j]
      , color="blue"
      , density=True
      )

    name = prefix + "regressed_%s" % labels[i]
    save_fig(fig, name, writer, epoch, outdir)

    fig.clf()

    ax = fig.add_subplot(111)
    ax.hist2d \
      ( targs
      , thesemus
      , bins=np.mgrid[bmin:bmax:51j]
      , norm=colors.LogNorm()
      )

    ax.plot([-10000, 10000], [-10000, 10000], color="red")
    ax.set_xlim(bmin, bmax)
    ax.set_ylim(bmin, bmax)

    name = prefix + "true_vs_regressed_%s" % labels[i]
    try:
      save_fig(fig, name, writer, epoch, outdir)
    except ValueError:
      pass

    fig.clf()


    if writer:
      writer.add_scalar(prefix + "meanresponse_%s" % labels[i], np.mean(resps), global_step=epoch)
      writer.add_scalar(prefix + "stdresponse_%s" % labels[i], std1(resps, 3), global_step=epoch)
      writer.add_scalar(prefix + "meandiff_%s" % labels[i], np.mean(diffs), global_step=epoch)
      writer.add_scalar(prefix + "stddiff_%s" % labels[i], np.std(diffs), global_step=epoch)
      writer.add_scalar(prefix + "meansigma_%s" % labels[i], np.mean(thesesigmas), global_step=epoch)
      writer.add_scalar(prefix + "meanpull_%s" % labels[i], np.mean(pulls), global_step=epoch)
      writer.add_scalar(prefix + "stdpull_%s" % labels[i], np.std(pulls), global_step=epoch)

    continue


  return
