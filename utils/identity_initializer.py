
'''tensorflow identity initializer, adapted from tensorflow official repo.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.init_ops import Initializer

class identity_initializer(Initializer):
  """Initializer that generates tensors initialized to identity matrix.
  """

  #TODO: for now, the function only works for 2-D matrix.
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return linalg_ops.eye(shape[0], shape[1], dtype=dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}