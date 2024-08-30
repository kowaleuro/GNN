# @title 2d Histogram function

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.pyplot as plt

def Create2d(mean,y_origin,y_predicted,bins,title,save: bool = False,saveName = ''):
  fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
  f1 = axs[0].hist2d(mean,y_origin,bins=bins, label = 'original',norm=LogNorm())
  axs[1].hist2d(mean,y_predicted,bins=bins, label = 'predicted',norm=LogNorm())

  axs[0].set_title('Rzeczywiste',loc = "left")
  axs[1].set_title('Aproksymowane',loc = "right")
  fig.suptitle(title)
  divider = make_axes_locatable(axs[1])
  cax = divider.append_axes("right", size="10%", pad=0.2)
  fig.colorbar(f1[3], cax=cax)
  fig.show()
  if save:
    fig.savefig(f'assets/{saveName}.pdf')

# Seaborn
def KDE(mean,y_origin,y_predicted,title,save: bool = False,saveName = ''):
  fig, ax =plt.subplots(1,2)
  sns.set_style("white")
  sns.kdeplot(x=mean,y=y_origin,cmap="Greens", fill=True,ax=ax[0])
  sns.kdeplot(x=mean,y=y_predicted,cmap="Reds", fill=True,ax=ax[1])
  ax[0].set_title('Rzeczywiste',loc = "center",y=0.9, va="top")
  ax[1].set_title('Aproksymowane',loc = "center",y=0.9, va="top")
  fig.suptitle(title)
  fig.show()
  if save:
    fig.savefig(f'assets/{saveName}.pdf')