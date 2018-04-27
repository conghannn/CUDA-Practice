import numpy
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import pycuda.driver as drv 

'''

config_file = "cuda_config_file"
profile_file = "cuda_profile_file"

drv.initialize_profiler(config_file,profile_file,drv.profiler_output_mode.CSV)

drv.start_profiler()





a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32)) 
print a_gpu.gpudata,"gpudata"
print a_gpu.shape,"shape"
print a_gpu.dtype,"dtype"
print a_gpu.size,"size"
print a_gpu.mem_size,"mem_size"
print a_gpu.nbytes,"nbytes"
print a_gpu.strides,"strides"
print a_gpu.flags,"flags"
print a_gpu.ptr,"ptr"
print a_gpu.__len__(), "__len__()"
ttt = a_gpu.ravel()
print ttt
ttt.set(10*numpy.random.randn(1,16).astype(numpy.float32))
print ttt
#a_doubled = (2*a_gpu).get()
#print a_doubled
#print a_gpu

drv.stop_profiler()
'''
a = drv.device_attribute.WARP_SIZE
print a


b = drv.Context.get_shared_config()
print b

d = drv.Device.count()
print d
c = drv.Device(0).compute_capability()
print c