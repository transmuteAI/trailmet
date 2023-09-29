- The following quantization configuration schema has been adopted throughout all quantizable modules to ensure compatibility with `torch.ao.quantization`.
    - **qscheme**: enum[per_tensor_affine, per_tensor_symmetric, per_channel_affine, per_channel_symmetric]
    - **dtype**: enum[qint8, quint8, qint32]
    - **scale**: type[float]
    - **zero_point**: type[int]
    - **quant_min**: type[int]
    - **quant_max**: type[int]
    
    for each quantized module `qscheme` is set according to the quantization algorithm and module under consideration, then `quant_min` and `quant_max` are determined based on the compression requirement and approximate precision importance for the given module. `dtype` is set such that it satisfies the given `quant_min` and `quant_max` range. `scale` and `zero_point` are finally determined dynamically during calibration using the applied algorithm.

- current implementation for deployment (x86 cpu only) stores model weights in `int8`/`int32` along with scale (`float32`) and zero_point (`int64`), effectively reducing required memory space (by about a factor of 4 for layer-wise granularity, and slightly lower for channel-wise granularity). Activations are quantized to `uint8` based on scaling factors determined during calibration (static during inference).

- when using the `x86 backend`, we need to use 7 bits instead of 8 bits. Make sure you reduce the range for the `quant_min`, `quant_max`, ie. if dtype is `torch.quint8`, we need to set `quant_min` to be 0 and `quant_max` to be 127 (255 / 2) and if dtype is `torch.qint8`, make sure to set `quant_min` to be -64 (-128 / 2) and `quant_max` to be 63 (127 / 2). This functionality is implemented and can be enabled by setting the configuration argument `reduce_range` to True. However, no need for this in `qnnpack backend`.