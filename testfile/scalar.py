from pycuda.reduction import ReductionKernel
import numpy
dot = ReductionKernel(dtype_out=numpy.float32, neutral="0",reduce_expr="a+b", map_expr="x[i]∗y[i]", arguments="const float ∗x, const float ∗y")
 
  
from pycuda.curandom import rand as curand
x = curand((1000*1000), dtype=numpy.float32)
y = curand((1000*1000), dtype=numpy.float32)
x_dot_y = dot(x, y).get()
x_dot_y_cpu = numpy.dot(x.get(), y.get())
print x_dot_y
print x_dot_y_cpu
