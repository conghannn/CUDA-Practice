import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy
import time

from pycuda.compiler import SourceModule

config_file = "cuda_config_file"
profile_file = "cuda_profile_file"
cuda.initialize_profiler(config_file, profile_file, cuda.profiler_output_mode.KEY_VALUE_PAIR)

cuda.start_profiler()

mod = SourceModule("""
__global__ void transpose(float *a, float *b, const unsigned int m, const unsigned int n)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < m)&&(Col < n)){
    b[Col * m + Row] = a[Row * n + Col];
  }
}
""")

m = 10000
n = 10000

a = numpy.random.randn(m, n)
a = a.astype(numpy.float32)
y_py = numpy.empty((n,m),a.dtype)

start = time.time()
y_py = numpy.transpose(a)
time1 = time.time() - start

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b_gpu = gpuarray.empty((n,m), a.dtype)

t_result = numpy.empty((n,m), a.dtype)


func = mod.get_function("transpose")
start = time.time()
func(a_gpu, b_gpu, numpy.uint32(m), numpy.uint32(n), block = (16,16,1), grid = ((numpy.uint32(n) - 1) / 16 + 1, (numpy.uint32(m) - 1) / 16 + 1, 1))
time2 = time.time() - start
t_result = b_gpu.get()

start = time.time()
time3 = time.time() - start

#cuda.memcpy_dtoh(t_result, b_gpu)
print time1
print time2
print time3
cuda.stop_profiler()

