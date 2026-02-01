import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantReLU
from brevitas.quant_tensor import QuantTensor
from brevitas.quant import Int8ActPerTensorFloat

# 1. Setup Layer - Left "normal" (no output_bit_width or output_quant)
q_conv = QuantConv2d(
    1, 1, 
    kernel_size=1, 
    bias=False, 
    weight_bit_width=4, 
    input_bit_width=4,
    return_quant_tensor=True,
    output_bit_width=8,           # Define the target width
    output_quant=Int8ActPerTensorFloat, # Define the quantization logic
)

with torch.no_grad():
    q_conv.weight.copy_(torch.tensor([[[[2.0]]]]))

# 2. Define Quantized ReLU
quant_relu = QuantReLU(bit_width=4, return_quant_tensor=True)

# 3. Create a NORMAL PyTorch Tensor
normal_input = torch.tensor([[[[-3.0], [4.0]]]], dtype=torch.float32)

# --- TRACKING METADATA ---

# Output of Conv
out_conv = q_conv(normal_input)

# REPAIR STEP: If bit_width is None, force it to 8 (or any target) 
if out_conv.bit_width is None:
    print("\n--- Repairing missing metadata in Conv output ---")
    out_conv = QuantTensor(
        value=out_conv.value,
        scale=out_conv.scale,
        zero_point=out_conv.zero_point,
        bit_width=8, # Manually assigning the expected bit-width
        signed=out_conv.signed,
        training=out_conv.training
    )

print("\n--- After Conv (with manual metadata repair) ---")
print(f"Type: {type(out_conv)}")
print(f"Has bit_width? {hasattr(out_conv, 'bit_width')}")
print(f"bit_width = {out_conv.bit_width}")

# Output of QuantReLU
# Note: QuantReLU will re-quantize this 8-bit input down to 4-bit
out_quant_relu = quant_relu(out_conv)

print("\n--- After QuantReLU ---")
print(f"Type: {type(out_quant_relu)}")
print(f"bit_width = {out_quant_relu.bit_width}")