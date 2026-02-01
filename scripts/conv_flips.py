import torch
from brevitas.nn import QuantConv2d
from brevitas.quant_tensor import QuantTensor

def flip_bits_logic(quant_tensor: QuantTensor, bit_index: int) -> QuantTensor:
    """Helper to flip bits on a QuantTensor while maintaining metadata."""
    # Ensure we are working with a QuantTensor
    if not isinstance(quant_tensor, QuantTensor):
        raise TypeError(f"Expected QuantTensor, got {type(quant_tensor)}")

    bw = int(quant_tensor.bit_width)
    is_signed = quant_tensor.signed
    
    # 1. Define range based on signedness
    if is_signed:
        min_int, max_int = -(2**(bw - 1)), 2**(bw - 1) - 1
    else:
        min_int, max_int = 0, 2**bw - 1

    # 2. Extract integers safely
    # (Value / Scale) + ZP = Integer Representation
    int_repr = (quant_tensor.value / quant_tensor.scale + quant_tensor.zero_point).round().to(torch.int32)
    
    # 3. XOR flip
    mask = 1 << bit_index
    flipped_int = int_repr ^ mask
    
    # 4. Hardware-style wrap around (simulating overflow/underflow)
    range_size = max_int - min_int + 1
    flipped_int = (flipped_int - min_int) % range_size + min_int

    # 5. Reconstruct as a new QuantTensor
    new_float_value = (flipped_int.float() - quant_tensor.zero_point) * quant_tensor.scale
    return QuantTensor(
        value=new_float_value, 
        scale=quant_tensor.scale, 
        zero_point=quant_tensor.zero_point,
        bit_width=bw, 
        signed=is_signed
    )

# --- SETUP ---
# Create a 4-bit Signed Conv layer that RETURNS a QuantTensor
q_conv = QuantConv2d(
    1, 1, kernel_size=1, bias=False, 
    weight_bit_width=4, 
    return_quant_tensor=True
)

# Set weight to 3.0 (Binary: 0011)
with torch.no_grad():
    q_conv.weight.copy_(torch.tensor([[[[3.0]]]]))

# Create input 2.0 (Binary: 0010)
input_qt = QuantTensor(
    value=torch.tensor([[[[2.0]]]]),
    scale=torch.tensor(1.0),
    zero_point=torch.tensor(0.0),
    bit_width=4,
    signed=True
)

# --- EXECUTION ---

print("--- INITIAL STATE ---")
print(f"Weight Int: {q_conv.quant_weight().int().item()}") # 3

# 1. Flip Weight MSB (Bit 3)
# 3 (0011) -> flip bit 3 -> 1011 (which is -5 in signed 4-bit)
weight_qt = q_conv.quant_weight()
flipped_weight_qt = flip_bits_logic(weight_qt, bit_index=3)

with torch.no_grad():
    q_conv.weight.copy_(flipped_weight_qt.value)

print(f"Flipped Weight Int: {q_conv.quant_weight().int().item()}") # -5

# 2. Run Forward Pass
# Calculation: Weight (-5) * Input (2) = -10
# Clamping: 4-bit signed range is [-8, 7]. -10 is clamped to -8.
output_dirty = q_conv(input_qt) 
print(f"Output before flip (Int): {output_dirty.int().item()}") # -8

# 3. Flip Output MSB (Bit 3)
# -8 (Binary: 1000) -> flip bit 3 -> 0000 (0)
final_output = flip_bits_logic(output_dirty, bit_index=3)

print("\n--- FINAL RESULT ---")
print(f"Final Output Int: {final_output.int().item()}") # 0