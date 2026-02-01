import torch
from brevitas.quant_tensor import QuantTensor

def flip_quant_tensor_bits(quant_tensor: QuantTensor, bit_index: int) -> QuantTensor:
    """
    Flips a specific bit in a QuantTensor and returns a new valid QuantTensor.
    Explicitly handles Float-to-Int conversion for bitwise operations.
    """
    bw = quant_tensor.bit_width
    is_signed = quant_tensor.signed
    
    # 1. Determine valid integer ranges for 4-bit (or any bit-width)
    if is_signed:
        min_int = -(2**(bw - 1))
        max_int = 2**(bw - 1) - 1
    else:
        min_int = 0
        max_int = 2**bw - 1

    # 2. Extract integers safely
    # We round first, then cast to .to(torch.int32) to support the ^ operator
    raw_int = (quant_tensor.value / quant_tensor.scale + quant_tensor.zero_point).round().to(torch.int32)
    
    # 3. Apply the Bit Flip using XOR
    # mask: bit 0 = 1, bit 1 = 2, bit 2 = 4, bit 3 = 8...
    mask = 1 << bit_index
    flipped_int = raw_int ^ mask
    
    # 4. Handle Hardware Wrap-around (Two's Complement Logic)
    # This ensures that flipping a bit doesn't result in an out-of-range value
    # for the specified bit-width.
    range_size = max_int - min_int + 1
    flipped_int = (flipped_int - min_int) % range_size + min_int

    # 5. Reconstruct the QuantTensor
    # Resulting float value = (Int - ZP) * Scale
    new_float_value = (flipped_int.float() - quant_tensor.zero_point) * quant_tensor.scale
    
    return QuantTensor(
        value=new_float_value,
        signed=is_signed,
        bit_width=bw,
        scale=quant_tensor.scale,
        zero_point=quant_tensor.zero_point,
        training=quant_tensor.training
    )

# --- Verification ---
if __name__ == "__main__":
    # Test Signed
    q_signed = QuantTensor(
        value=torch.tensor([-7.0, -1.0, 0.0, 1.0, 6.0]),
        scale=torch.tensor(1.0),
        zero_point=torch.tensor(0.0),
        bit_width=4,
        signed=True
    )
    res_signed = flip_quant_tensor_bits(q_signed, bit_index=3)
    print(f"Signed Original: {q_signed.int().tolist()}")
    print(f"Signed Flipped:  {res_signed.int().tolist()}")

    # Test Unsigned
    q_unsigned = QuantTensor(
        value=torch.tensor([0.0, 1.0, 7.0, 8.0, 15.0]),
        scale=torch.tensor(1.0),
        zero_point=torch.tensor(0.0),
        bit_width=4,
        signed=False
    )
    res_unsigned = flip_quant_tensor_bits(q_unsigned, bit_index=3)
    print(f"\nUnsigned Original: {q_unsigned.int().tolist()}")
    print(f"Unsigned Flipped:  {res_unsigned.int().tolist()}")