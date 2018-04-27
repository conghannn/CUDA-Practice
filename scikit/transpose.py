#!/usr/bin/env python
 
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
import pycuda.autoinit
import string
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

from skcuda.cula import *

culinalg.init()

N = range(1, 30)
time_cpu = []
time_linalg = []
time_cula = []
for i in N:
	t = np.float32
	n = i * 32
    
	a = np.asarray(np.random.rand(n,n), t)


	start = time.time()
	c = np.transpose(a)
	time_cpu.append(time.time() - start)

	a_gpu = gpuarray.to_gpu(a)

	start = time.time()
	c_gpu = culinalg.transpose(a_gpu)
	time_linalg.append(time.time() - start)

	
	
	a_gpu2 = gpuarray.to_gpu(a)
	cula_result = gpuarray.empty((n, n), np.float32)

	#culaGetVersion
	'''
	culaInitialize
	
	start = time.time()
	culaDeviceSgeTranspose(n, n, a_gpu2.gpudata, n, cula_result.gpudata, n)
	time_cula.append(time.time() - start)
	
	culaShutdown
	'''



MAKE_PLOT = False
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_linalg,'g')
  #plt.plot(N, time_cula,'b')
  plt.legend(['cpu time', 'linalg time'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('transpose_plot.png')