#!/usr/bin/env python
import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler, gpuarray, tools
import matplotlib as mpl
mpl.use('agg')
import time



def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

mod = SourceModule("""
__global__ void naive_hist(unsigned char *img, unsigned int *bins, int P, int size )
{
    unsigned w = blockDim.x * gridDim.x;
    unsigned int i = (blockIdx.y * blockDim.y + threadIdx.x) * w + threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int k;
    unsigned int bins_loc[256];
    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (k=0; k<P; k++){
        if((i*P+k)<size)
            ++bins_loc[img[i*P+k]];
    }
    for (k=0; k<256; k++)
        atomicAdd(&bins[k], bins_loc[k]);
}
__global__ void opt_hist(unsigned char *buffer, unsigned int *histo, int size)
{
    __shared__ unsigned int private_histo[256];
    if (threadIdx.x < 256){
        private_histo[threadIdx.x]= 0;}
    __syncthreads();
    int i = threadIdx.x + blockIdx.x *blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size)
    {
        atomicAdd(&(private_histo[buffer[i]]), 1);
        i += stride;
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        atomicAdd(&(histo[threadIdx.x]),private_histo[threadIdx.x] );
    }
}
""")



time_naive = []
time_opt = []

# compare the time of opencl naive version and optimized version:
def optimized_hist(img, R):
    K = 16
    opt_hist = mod.get_function("opt_hist")
    img_buf = gpuarray.to_gpu(img)
    hist_buf = gpuarray.zeros(256, np.int32)

    start = time.time()
    opt_hist(img_buf, hist_buf, np.int32(R), block=(1024, 1, 1), grid=(K, 1, 1))
    time2.append(time.time() - start)

def naive_hist(img, R):
    l = np.sqrt(R).astype(np.int32)
    l = int(l) + 1
    N = 32
    naive_hist = mod.get_function("naive_hist")
    hist_buf1 = gpuarray.zeros(256, np.int32)
    img_buf = gpuarray.to_gpu(img)
    
    start = time.time()
    naive_hist(img_buf, hist_buf1, np.int32(N), np.int32(R), block = (16, 16, 1), grid=((l - 1) // (16 * 32) + 1, (l - 1) // (16 * 32) + 1, 1))
    time1.append(time.time() - start)

# different image of size 1MB,10MB,50MB,100MB,500MB
'''
img1 = np.memmap("/opt/data/random.dat", dtype='uint8', mode='r', shape=(1024,1024))
R1 = 1024*1024
img2 = np.memmap("/opt/data/random.dat", dtype='uint8', mode='r', shape=(3100,3100))
R2 = 3100*3100
img3 = np.memmap("/opt/data/random.dat", dtype='uint8', mode='r', shape=(7071,7071))
R3 = 7071*7071
img4 = np.memmap("/opt/data/random.dat", dtype='uint8', mode='r', shape=(10240,10240))
R4 = 10240*10240
img5 = np.memmap("/opt/data/random.dat", dtype='uint8', mode='r', shape=(22879,22879))
R5 = 22879*22879
'''


# compare the result of optimized opencl version with python version with image size of
# 1MB
TEST = False
if TEST:
    img1 = np.random.randint(255, size = (1024, 1024)).astype(np.uint8)
    R1 = 1024 * 1024
    img2 = np.random.randint(255, size = (3100, 3100)).astype(np.uint8)
    R2 = 3100 * 3100
    img3 = np.random.randint(255, size = (7071, 7071)).astype(np.uint8)
    R3 = 7071 * 7071
    img4 = np.random.randint(255, size = (10240, 10240)).astype(np.uint8)
    R4 = 10240 * 10240
    img5 = np.random.randint(255, size = (22879, 22879)).astype(np.uint8)
    R5 = 22879 * 22879
    opt_hist = mod.get_function("opt_hist")
    img_buf = gpuarray.to_gpu(img1)
    hist_buf = gpuarray.zeros(256, np.int32)
    opt_hist(img_buf, hist_buf, np.int32(R1), block=(256, 1, 1), grid=(16, 1, 1))
    print "whether the optimized version's result is right:",np.allclose(hist_buf.get(),hist(img1))
    # compare time for different size(1MB,10MB,50MB,100MB,500MB)
    start = time.time()
    optimized_hist(img1,R1)
    time_opt.append(time.time() - start)
    start = time.time()
    optimized_hist(img2,R2)
    time_opt.append(time.time() - start)
    start = time.time()
    optimized_hist(img3,R3)
    time_opt.append(time.time() - start)
    start = time.time()
    optimized_hist(img4,R4)
    time_opt.append(time.time() - start)
    start = time.time()
    optimized_hist(img5,R5)
    time_opt.append(time.time() - start)

    naive_hist(img2,R2)

    start = time.time()
    naive_hist(img1,R1)
    time_naive.append(time.time() - start)
    start = time.time()
    naive_hist(img2,R2)
    time_naive.append(time.time() - start)
    start = time.time()
    naive_hist(img3,R3)
    time_naive.append(time.time() - start)
    start = time.time()
    naive_hist(img4,R4)
    time_naive.append(time.time() - start)
    start = time.time()
    naive_hist(img5,R5)
    time_naive.append(time.time() - start)
    # print time
    print "naive version time:",time_naive
    print "optimized version time:",time_opt
    ratio = []
    for i in xrange(5):
        ratio.append(time_naive[i]/time_opt[i])

    print "ratio:time_naive/time_opt",ratio

'''


# plot
N = range(1,6)
plt.figure(1)
plt.plot(N, time_naive,label='naive_time')
plt.plot(N, time_opt,label='optimized_time')
plt.xlabel('image size')
plt.ylabel('time/second')
plt.gca().set_xlim((0, max(N)))
plt.legend()
plt.savefig('cuda_histogram.png')
plt.figure(2)
plt.plot(N, ratio,label='ratio')
plt.xlabel('image size')
plt.ylabel('time_naive:time_opt')
plt.gca().set_xlim((0, max(N)))
plt.legend()
plt.savefig('cuda_ratio.png')
'''


N = range(1, 300)
time1 = []
time2 = []
for i in N:
    print i
    size = i * 128
    img = np.random.randint(255, size = (size, size)).astype(np.uint8)
    R = size * size
    naive_hist(img, R)
    optimized_hist(img, R)
ratio = []


for i in xrange(len(time1)):
    ratio.append(time1[i] / time2[i])


MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time1,'g')
  plt.plot(N, time2,'b')
  plt.legend(['gpu naive', 'gpu opt'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(312)
  plt.plot(N, ratio,'r')
  plt.legend(['ratio'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('ratio')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')



