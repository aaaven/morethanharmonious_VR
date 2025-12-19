import torch
import torch.nn as nn
import copy

import pytest
import unittest

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


is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4096, 14336, bias=True)
        self.w3 = nn.Linear(4096, 14336, bias=True)
        self.w2 = nn.Linear(14336, 4096, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.w1(x)
        y3 = self.w3(x)
        return self.w2(F.silu(y1 * y3))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()



class TestHPTrainToFP8LinearInference:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        with patch("torch._dynamo.config.cache_size_limit", 20):
            yield

    def base_test_mlp_transform(self, base_mlp, quantized_mlp, input_tensor):
        with torch.no_grad():
            base_output = base_mlp(input_tensor)
            transformed_output = quantized_mlp(input_tensor)

        # Compute and check SQNR
        sqnr = compute_error(base_output, transformed_output)
        assert sqnr.item() > 20, f"SQNR is too low: {sqnr.item()} dB"

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_dynamic_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        dynamic_fp8_mlp = copy.deepcopy(original_mlp)

        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(dynamic_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_dynamic_fp8_mlp = torch.compile(
            dynamic_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            original_mlp, dynamic_fp8_mlp, input_tensor
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_dynamic_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_static_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(
            ActivationCasting.STATIC,
            static_quantization_scale=torch.tensor(
                [1.0], device="cuda", dtype=torch.float32
            ),
        )
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp,
            backend=compile_backend,
            fullgraph=True,
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            original_mlp, static_fp8_mlp, input_tensor
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_weight_only_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(ActivationCasting.WEIGHT_ONLY)
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )




if __name__ == '__main__':

    test = TestHPTrainToFP8LinearInference()
    test.test_static_fp8_mlp("eager", torch.bfloat16)
