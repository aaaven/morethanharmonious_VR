import copy

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    unwrap_tensor_subclass,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export._trace import _export as _export_private
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_tensor import Float8Tensor
from torchao.float8.float8_utils import compute_error
from torchao.float8.inference import (
    ActivationCasting,
    Float8InferenceLinear,
    QuantConfig,
    quantize_to_float8,
)

from pipeline import pipe_reduced
import torch

unet_kwargs = torch.load('unet_kwargs.pt', weights_only=True)

import time

with torch.inference_mode():

    for i in range(5):
        start_time = time.time()
        res = pipe_reduced.unet(**unet_kwargs)
        torch.cuda.synchronize()
        print(f"time: {time.time() - start_time}")
        del res
        torch.cuda.empty_cache()

    unet = copy.deepcopy(pipe_reduced.unet)

    quant_config = QuantConfig(
        ActivationCasting.STATIC,
        static_quantization_scale=torch.tensor(
            [1.0], device="cuda", dtype=torch.bfloat16
        )
    )
    quantize_to_float8(unet, quant_config)


    print('quantize to float8')

    for i in range(5):
        start_time = time.time()
        res = unet(**unet_kwargs)
        torch.cuda.synchronize()
        print(f"time: {time.time() - start_time}")
        del res
        torch.cuda.empty_cache()


    compile_backend = "eager"
    compile_backend = "inductor"
    compiled_original_unet = torch.compile(pipe_reduced.unet, backend=compile_backend, fullgraph=True)
    compiled_dynamic_fp8_unet = torch.compile(unet, backend=compile_backend, fullgraph=True)


    print('compiled origin unet')
    for i in range(5):
        start_time = time.time()
        res = compiled_original_unet(**unet_kwargs)
        torch.cuda.synchronize()
        print(f"time: {time.time() - start_time}")
        del res
        torch.cuda.empty_cache()
    

    print('compiled quantized unet')
    
    for i in range(5):
        start_time = time.time()
        res = compiled_dynamic_fp8_unet(**unet_kwargs)
        torch.cuda.synchronize()
        print(f"time: {time.time() - start_time}")
        del res
        torch.cuda.empty_cache()

