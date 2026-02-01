import torch
from brevitas.nn import QuantConv2d, QuantReLU

# 1. Setup Layer: 4-bit weights
q_conv = QuantConv2d(
    1, 1, kernel_size=2, bias=False, weight_bit_width=4, return_quant_tensor=True, output_quant=
)

# Set weights to 7 (Max 4-bit signed)
with torch.no_grad():
    q_conv.weight.copy_(torch.full((1, 1, 2, 2), 7.0))

# 2. Setup QuantReLU: 4-bit
q_relu = QuantReLU(bit_width=4, return_quant_tensor=True)

# 3. Create Input: 2x2 patch of 7 (Max 4-bit signed)
# We define this as a QuantTensor so we can track the scales
input_qt = torch.full((1, 1, 2, 2), 7.0)

# --- THE VERIFICATION ---

# Step 1: Convolution
out_conv = q_conv(input_qt)

print("--- 1. AFTER CONV (High-Precision Accumulator) ---")
print(f"Metadata bit_width: {out_conv.bit_width}")
print(f"Floating Value:     {out_conv.value.item()}")

# Manual Integer Calculation
# Because out_conv.scale is None, we use the property: S_out = S_input * S_weight
s_weight = q_conv.quant_weight().scale
# Assuming input scale was 1.0 (default for simple tensors)
s_input = 1.0
s_out = s_weight * s_input

manual_int = (out_conv.value / s_out).round().int()
print(f"Manual Integer:     {manual_int.item()}")
print(f"Is 196 representable in 4-bit signed (-8 to 7)? {manual_int.item() <= 7}")

# Step 2: ReLU
out_relu = q_relu(out_conv)

print("\n--- 2. AFTER RELU (Re-quantized to 4-bit) ---")
print(f"Metadata bit_width: {out_relu.bit_width}")
print(f"Floating Value:     {out_relu.value.item()}")
print(f"Valid Integer (.int()): {out_relu.int().item()}")
