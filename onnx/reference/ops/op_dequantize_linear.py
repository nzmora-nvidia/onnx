# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class DequantizeLinear(OpRun):
    def _run(self, *args, axis=None, dtype=None):  # type: ignore
        if len(args[1].shape) > 1:
            raise RuntimeError("Input 2 must be a vector or a number.")

        # Default output is float.
        output_dtype = dtype or np.float32
        # Numpy doesn't support BFloat16.
        if output_dtype not in (np.float32, np.float16):
            raise RuntimeError(
                f"Output data type {output_dtype} is not supported in DequantizeLinear."
            )
        x_scale = args[2]
        if len(x_scale.shape) > 0 and x_scale.size == 1:
            x_scale = x_scale[0]
        if len(args) > 2:
            if x_scale.dtype != np.float32:
                raise RuntimeError(
                    f"Scale data type {x_scale.dtype} is not legal in DequantizeLinear."
                )
            if len(x_scale.shape) > 0:
                new_shape = [1 for s in args[0].shape]
                new_shape[axis] = len(x_scale)
                x = args[0].astype(np.float32) - x_scale.reshape(new_shape)
                y = x * args[1].reshape(new_shape)
            else:
                x = args[0].astype(np.float32) - x_scale
                y = x * args[1]
        elif len(args[1].shape) > 0:
            new_shape = [1 for s in args[0].shape]
            new_shape[axis] = len(x_scale)
            y = args[0].astype(np.float32) * x_scale.reshape(new_shape)
        else:
            y = args[0].astype(np.float32) * x_scale
        return (y.astype(output_dtype),)
