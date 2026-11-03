from time import perf_counter

import numpy as np


def _prepare_conv(x, w, b, conv_param):
  """Validate NCHW convolution inputs and prepare shared inference data.

  Returns (x_padded, stride, output_shape), where output_shape is
  (N, F, out_h, out_w). Padding is symmetric and does not modify x.
  Incomplete windows at the right/bottom are discarded.
  """
  if x.ndim != 4 or w.ndim != 4:
    raise ValueError("x and w must be 4-dimensional NCHW arrays")

  N, C, H, W = x.shape
  F, filter_channels, HH, WW = w.shape
  if C != filter_channels:
    raise ValueError("input and filter channel counts must match")
  if b.shape != (F,):
    raise ValueError("b must have shape (F,)")
  if min(C, H, W, F, HH, WW) <= 0:
    raise ValueError("channel, spatial, and filter dimensions must be positive")

  stride, pad = conv_param["stride"], conv_param["pad"]
  if (
    isinstance(stride, (bool, np.bool_))
    or not isinstance(stride, (int, np.integer))
    or stride <= 0
  ):
    raise ValueError("stride must be a positive integer")
  if (
    isinstance(pad, (bool, np.bool_))
    or not isinstance(pad, (int, np.integer))
    or pad < 0
  ):
    raise ValueError("pad must be a non-negative integer")
  if HH > H + 2 * pad or WW > W + 2 * pad:
    raise ValueError("filter must fit within the padded input")

  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
  out_h = (H + 2 * pad - HH) // stride + 1
  out_w = (W + 2 * pad - WW) // stride + 1

  return x_padded, stride, (N, F, out_h, out_w)


def conv_forward_native(x, w, b, conv_param):
  """Run NCHW convolution inference with explicit loops.

  x: input of shape (N, C, H, W).
  w: filters of shape (F, C, HH, WW), applied without flipping the kernel.
  b: biases of shape (F,).
  conv_param: integer 'stride' > 0 and symmetric zero 'pad' >= 0.

  Returns a contiguous array of shape (N, F, out_h, out_w), without a
  backward-pass cache. Incomplete windows at the right/bottom are discarded.
  The input arrays are not modified.
  """
  x_padded, stride, output_shape = _prepare_conv(x, w, b, conv_param)
  N, F, out_h, out_w = output_shape
  _, _, HH, WW = w.shape
  out = np.zeros(output_shape, dtype=np.result_type(x, w, b))

  for n in range(N):
    for f in range(F):
      for i in range(out_h):
        for j in range(out_w):
          h_start, w_start = i * stride, j * stride
          window = x_padded[n, :, h_start : h_start + HH, w_start : w_start + WW]
          out[n, f, i, j] = np.sum(window * w[f]) + b[f]
  return out


def conv_forward_fast(x, w, b, conv_param):
  """Run NCHW convolution inference using im2col and matrix multiplication.

  x: input of shape (N, C, H, W).
  w: filters of shape (F, C, HH, WW), applied without flipping the kernel.
  b: biases of shape (F,).
  conv_param: integer 'stride' > 0 and symmetric zero 'pad' >= 0.

  Returns a contiguous array of shape (N, F, out_h, out_w), without a
  backward-pass cache. Incomplete windows at the right/bottom are discarded.
  The input arrays are not modified.
  """
  x_padded, stride, output_shape = _prepare_conv(x, w, b, conv_param)
  N, F, out_h, out_w = output_shape
  _, C, HH, WW = w.shape

  # Use actual byte strides so non-contiguous inputs are handled correctly.
  sN, sC, sH, sW = x_padded.strides
  x_stride = np.lib.stride_tricks.as_strided(
    x_padded,
    shape=(C, HH, WW, N, out_h, out_w),
    strides=(sC, sH, sW, sN, stride * sH, stride * sW),
    writeable=False,
  )
  x_cols = np.ascontiguousarray(x_stride).reshape(
    C * HH * WW, N * out_h * out_w
  )

  res = w.reshape(F, C * HH * WW).dot(x_cols) + b.reshape(F, 1)
  out = res.reshape(F, N, out_h, out_w).transpose(1, 0, 2, 3)
  return np.ascontiguousarray(out)


def main():
  """Compare inference outputs and average runtimes on identical inputs."""
  rng = np.random.default_rng(42)
  x = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
  w = rng.standard_normal((16, 3, 3, 3)).astype(np.float32)
  b = rng.standard_normal(16).astype(np.float32)
  conv_param = {"stride": 1, "pad": 1}

  # Check correctness and warm up both implementations before timing.
  out_native = conv_forward_native(x, w, b, conv_param)
  out_fast = conv_forward_fast(x, w, b, conv_param)
  np.testing.assert_allclose(out_fast, out_native, rtol=1e-4, atol=1e-5)
  print(f"Input: {x.shape}, filters: {w.shape}, output: {out_fast.shape}")
  print("Outputs match (rtol=1e-4, atol=1e-5)")
  print(f"Maximum absolute error: {np.max(np.abs(out_fast - out_native)):.6g}")

  # Time complete calls, including shared preparation and convolution.
  repeats = 10
  native_times, fast_times = [], []
  for _ in range(repeats):
    start = perf_counter()
    conv_forward_native(x, w, b, conv_param)
    native_times.append(perf_counter() - start)

    start = perf_counter()
    conv_forward_fast(x, w, b, conv_param)
    fast_times.append(perf_counter() - start)

  native_mean = np.mean(native_times)
  fast_mean = np.mean(fast_times)
  print(f"Average time over {repeats} runs (after warmup):")
  print(f"  conv_forward_native: {native_mean * 1000:.3f} ms")
  print(f"  conv_forward_fast:  {fast_mean * 1000:.3f} ms")
  print(f"Speedup (native / fast): {native_mean / fast_mean:.2f}x")


if __name__ == "__main__":
  main()
