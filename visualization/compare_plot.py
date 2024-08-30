# @title Original vs predicted

import matplotlib.pyplot as plt

def plot_results(original, predicted, title, save=False, sampleNo=0,x_label='X',y_label='Y'):
  if sampleNo == 0:
    sampleNo = len(original)
  plt.plot(original[0:sampleNo], color='blue', label='Oryginalne')
  plt.plot(predicted[0:sampleNo], color='red', label='Aproksymowane')
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc='upper left')
  if save:
    plt.savefig('assets/Original vs Predicted.pdf')
  plt.show()