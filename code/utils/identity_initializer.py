from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import math

# from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
# from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.init_ops import Initializer

class identity_initializer(Initializer):
  """Initializer that generates tensors initialized to 0."""

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return linalg_ops.eye(shape[0], shape[1], dtype=dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}