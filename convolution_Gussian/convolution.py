#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import string
import time

KERNEL_RADIUS = 8
UNROLL_INNER_LOOP = True
KERNEL_W = 2 * KERNEL_RADIUS + 1
ROW_TILE_W = 128
KERNEL_RADIUS_ALIGNED = 16
COLUMN_TILE_W = 16
COLUMN_TILE_H = 48

Template = '''
#include <stdio.h>
#define IMUL(a, b) __mul24(a, b)
#define KERNEL_RADIUS $KERNEL_RADIUS
#define KERNEL_W $KERNEL_W
__device__ __constant__ float d_Kernel_rows[KERNEL_W];
__device__ __constant__ float d_Kernel_columns[KERNEL_W];

#define            ROW_TILE_W  $ROW_TILE_W
#define KERNEL_RADIUS_ALIGNED  $KERNEL_RADIUS_ALIGNED

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W $COLUMN_TILE_W
#define COLUMN_TILE_H $COLUMN_TILE_H

__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    const int          rowStart = IMUL(blockIdx.y, dataW);

    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;

    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] =
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    __syncthreads();
    
    const int writePos = tileStart + threadIdx.x;
    
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel_rows[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template(
    'sum += data[smemPos + $k] * d_Kernel_rows[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)


Template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
Template += '''
        d_Result[rowStart + writePos] = sum;
    }
}

__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    __shared__ float data[COLUMN_TILE_W *
    (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] =
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ?
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    __syncthreads();

    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + IMUL(k, COLUMN_TILE_W)] *
            d_Kernel_columns[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template('sum += data[smemPos + IMUL($k, COLUMN_TILE_W)] * d_Kernel_columns[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

Template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
Template += '''
        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}






__global__ void convolutionRowGPU_roll(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    const int          rowStart = IMUL(blockIdx.y, dataW);

    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;

    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] =
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    __syncthreads();
    
    const int writePos = tileStart + threadIdx.x;
    
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel_rows[KERNEL_RADIUS - k];
'''


Template += originalLoop
Template += '''
        d_Result[rowStart + writePos] = sum;
    }
}

__global__ void convolutionColumnGPU_roll(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    __shared__ float data[COLUMN_TILE_W *
    (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] =
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ?
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    __syncthreads();

    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + IMUL(k, COLUMN_TILE_W)] *
            d_Kernel_columns[KERNEL_RADIUS - k];
'''


Template += originalLoop
Template += '''
        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}




'''

template = string.Template(Template)
code = template.substitute(KERNEL_RADIUS = KERNEL_RADIUS,
                           KERNEL_W = KERNEL_W,
                           COLUMN_TILE_H=COLUMN_TILE_H,
                           COLUMN_TILE_W=COLUMN_TILE_W,
                           ROW_TILE_W=ROW_TILE_W,
                           KERNEL_RADIUS_ALIGNED=KERNEL_RADIUS_ALIGNED)

module = SourceModule(code)
convolutionRowGPU = module.get_function('convolutionRowGPU')
convolutionColumnGPU = module.get_function('convolutionColumnGPU')
d_Kernel_rows = module.get_global('d_Kernel_rows')[0]
d_Kernel_columns = module.get_global('d_Kernel_columns')[0]


# Helper functions for computing alignment...
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

def iDivDown(a, b):
    # Round a / b to nearest lower integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a / b;

def iAlignUp(a, b):
    # Align a to nearest higher multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a - a % b + b) if (a % b != 0) else a

def iAlignDown(a, b):
    # Align a to nearest lower multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a - a % b

def gaussian_kernel(width = KERNEL_W, sigma = 4.0):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    sigma = numpy.float32(sigma)
    filterx = x*x / (2 * sigma * sigma)
    filterx = numpy.exp(-1 * filterx)
    assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    filterx /= filterx.sum()
    return filterx

def derivative_of_gaussian_kernel(width = KERNEL_W, sigma = 4):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)

    filterx = gaussian_kernel(width,  sigma)
    filterx *= x

    scale = (x * filterx).sum()
    filterx /= scale

    filterx *= -1.0
    return filterx

def test_derivative_of_gaussian_kernel():
    width = 20
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filter * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'
    width = 19
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filterx * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'

def convolution_cuda(sourceImage,  filterx,  filtery):
    # Perform separable convolution on sourceImage using CUDA.
    # Operates on floating point images with row-major storage.
    destImage = sourceImage.copy()
    assert sourceImage.dtype == 'float32',  'source image must be float32'
    (imageHeight,  imageWidth) = sourceImage.shape
    assert filterx.shape == filtery.shape == (KERNEL_W, ) ,  'Kernel is compiled for a different kernel size! Try changing KERNEL_W'
    filterx = numpy.float32(filterx)
    filtery = numpy.float32(filtery)
    DATA_W = iAlignUp(imageWidth, 16);
    DATA_H = imageHeight;
    BYTES_PER_WORD = 4;  # 4 for float32
    DATA_SIZE = DATA_W * DATA_H * BYTES_PER_WORD;
    KERNEL_SIZE = KERNEL_W * BYTES_PER_WORD;
    # Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    cuda.memcpy_htod(d_Kernel_rows,  filterx) # The kernel goes into constant memory via a symbol defined in the kernel
    cuda.memcpy_htod(d_Kernel_columns,  filtery)
    # Call the kernels for convolution in each direction.
    blockGridRows = (iDivUp(DATA_W, ROW_TILE_W), DATA_H)
    blockGridColumns = (iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H))
    threadBlockRows = (KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS, 1, 1)
    threadBlockColumns = (COLUMN_TILE_W, 8, 1)
    DATA_H = numpy.int32(DATA_H)
    DATA_W = numpy.int32(DATA_W)
    grid_rows = tuple([int(e) for e in blockGridRows])
    block_rows = tuple([int(e) for e in threadBlockRows])
    grid_cols = tuple([int(e) for e in blockGridColumns])
    block_cols = tuple([int(e) for e in threadBlockColumns])
    convolutionRowGPU(intermediateImage_gpu,  sourceImage_gpu,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU(destImage_gpu,  intermediateImage_gpu,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)

    # Pull the data back from the GPU.
    cuda.memcpy_dtoh(destImage,  destImage_gpu)
    return destImage

def test_convolution_cuda():
    # Test the convolution kernel.
    # Generate or load a test image
    original = numpy.random.rand(2,  6) * 255
    original = numpy.float32(original)
    # You probably want to display the image using the tool of your choice here.
    filterx = gaussian_kernel()
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filterx)
    print destImage
    # You probably wand to display the result image using the tool of your choice here.
    print 'Done running the convolution kernel!'

#if __name__ == '__main__':
    #test_convolution_cuda()
    #test_derivative_of_gaussian_kernel()
    #boo = raw_input('Pausing so you can look at results... <Enter> to finish...')





convolutionRowGPU_roll = module.get_function('convolutionRowGPU_roll')
convolutionColumnGPU_roll = module.get_function('convolutionColumnGPU_roll')



N = range(1, 300)
time_roll = []
time_unroll = []

filterx = gaussian_kernel()
filtery = filterx


for i in N:
    size = i * 32
    sourceImage = (numpy.random.rand(size,  size) * 255).astype(numpy.float32)


    destImage = sourceImage.copy()
    assert sourceImage.dtype == 'float32',  'source image must be float32'
    (imageHeight,  imageWidth) = sourceImage.shape
    assert filterx.shape == filtery.shape == (KERNEL_W, ) ,  'Kernel is compiled for a different kernel size! Try changing KERNEL_W'
    filterx = numpy.float32(filterx)
    filtery = numpy.float32(filtery)
    DATA_W = iAlignUp(imageWidth, 16);
    DATA_H = imageHeight;
    BYTES_PER_WORD = 4;  # 4 for float32
    DATA_SIZE = DATA_W * DATA_H * BYTES_PER_WORD;
    KERNEL_SIZE = KERNEL_W * BYTES_PER_WORD;
    # Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    cuda.memcpy_htod(d_Kernel_rows,  filterx) # The kernel goes into constant memory via a symbol defined in the kernel
    cuda.memcpy_htod(d_Kernel_columns,  filtery)

    # Call the kernels for convolution in each direction.
    blockGridRows = (iDivUp(DATA_W, ROW_TILE_W), DATA_H)
    blockGridColumns = (iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H))
    threadBlockRows = (KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS, 1, 1)
    threadBlockColumns = (COLUMN_TILE_W, 8, 1)
    DATA_H = numpy.int32(DATA_H)
    DATA_W = numpy.int32(DATA_W)
    grid_rows = tuple([int(e) for e in blockGridRows])
    block_rows = tuple([int(e) for e in threadBlockRows])
    grid_cols = tuple([int(e) for e in blockGridColumns])
    block_cols = tuple([int(e) for e in threadBlockColumns])

    start = time.time()
    convolutionRowGPU_roll(intermediateImage_gpu,  sourceImage_gpu,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU_roll(destImage_gpu,  intermediateImage_gpu,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)
    time_roll.append(time.time() - start)


    destImage_gpu_unroll = cuda.mem_alloc_like(destImage)
    sourceImage_gpu_unroll = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu_unroll = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu_unroll, sourceImage)

    start = time.time()
    convolutionRowGPU(intermediateImage_gpu_unroll,  sourceImage_gpu_unroll,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU(destImage_gpu_unroll,  intermediateImage_gpu_unroll,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)
    time_unroll.append(time.time() - start)



MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(211)
  plt.plot(N, time_roll,'r')
  plt.plot(N, time_unroll,'g')
  plt.legend(['roll time', 'unroll time'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')














