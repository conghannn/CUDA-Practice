#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import string

template = '''
#include <stdio.h>
__global__ void convolutionRowGPU(float *d_Result, float *d_Data, int dataW, int dataH)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		float* a = (float*)((char*)d_Result + 0 * 512);
		printf("%f", a[10]);
		printf("%f", d_Result[128 + 2]);
	}
}
'''

#template = string.Template(template)
module = SourceModule(template)
func = module.get_function('convolutionRowGPU')

original = numpy.random.rand(2,  7) * 255
original = numpy.float32(original)

print original
'''
destImage_gpu = cuda.mem_alloc_like(original)
sourceImage_gpu = cuda.mem_alloc_like(original)
intermediateImage_gpu = cuda.mem_alloc_like(original)
'''
destImage_gpu, pit = cuda.mem_alloc_pitch(7 * 4, 2 , numpy.dtype(numpy.float32).itemsize) 
sourceImage_gpu, pit2 = cuda.mem_alloc_pitch(7 * 4, 2, numpy.dtype(numpy.float32).itemsize)
print pit, pit2

#cuda.memcpy_htod(sourceImage_gpu, original)
#cuda.memcpy_htod(destImage_gpu, original)
copy = cuda.Memcpy2D()
copy.set_src_host(original)
copy.set_dst_device(destImage_gpu)
copy.height = 2
copy.width_in_bytes = 7 * 4
copy.src_pitch = 7 * 4
copy.dst_pitch = 128 * 4
copy(aligned = True)

destImage = original.copy()


func(destImage_gpu, sourceImage_gpu, numpy.int32(10), numpy.int32(2), block = (10, 1, 1), grid = (1, 1, 1))

#cuda.memcpy_dtoh(destImage,  destImage_gpu)

copy = cuda.Memcpy2D()
copy.set_src_device(destImage_gpu)
copy.set_dst_host(destImage)
copy.height = 2
copy.width_in_bytes = 7 * 4
copy.src_pitch = 128 * 4
copy.dst_pitch = 7 * 4
copy(aligned=True)



print "\n"
print destImage
