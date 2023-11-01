#!/usr/bin/env python3

import struct
from typing import BinaryIO

# Generates a random GIF file into `out` using the
# random number generator `rng` (/dev/urandom)
def generate_random_gif(
        rng: BinaryIO, out: BinaryIO
    ):
    # 1. Header
    header = b'GIF89a'
    
    # 2. Logical Screen Descriptor
    # Random width and height (using two bytes for each)
    width, height = struct.unpack('>HH', rng.read(4))
    
    # Other LSD fields: Packed fields (using a random byte for variety), Background color index, Pixel aspect ratio
    lsd = struct.pack('>HHBHB', width, height, rng.read(1)[0], rng.read(1)[0], rng.read(1)[0])
    
    # 3. Global Color Table (let's make it have random size, capped at 256 colors, and fully random)
    num_colors = (rng.read(1)[0] % 256) or 1
    color_table = rng.read(3 * num_colors)  # Random colors
    
    # 4. Graphics Control Extension (optional)
    gce = struct.pack('>BBBHB', 0x21, 0xF9, 0x04, rng.read(1)[0], rng.read(1)[0])
    
    # 5. Image Descriptor
    left, top = struct.unpack('>HH', rng.read(4))
    width, height = struct.unpack('>HH', rng.read(4))
    image_descriptor = struct.pack('>BHHHHB', 0x2C, left, top, width, height, rng.read(1)[0])
    
    # 6. Image Data (randomly sized, capped at 255 bytes per sub-block)
    lzw_minimum_code_size = (rng.read(1)[0] % 8) or 1
    
    blocks = []
    total_blocks_size = rng.read(1)[0]  # Make it reasonably small
    for _ in range(total_blocks_size):
        block_size = rng.read(1)[0]  # Each block can be up to 255 bytes
        blocks.append(bytes([block_size]) + rng.read(block_size))
    image_data = bytes([lzw_minimum_code_size]) + b''.join(blocks) + b'\x00'  # Ending with a block size of 0
    
    # 7. Trailer
    trailer = b'\x3B'
    
    # Write everything to the output file
    out.write(header + lsd + color_table + gce + image_descriptor + image_data + trailer)
