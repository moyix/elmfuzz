#!/usr/bin/env python3

import struct
from typing import BinaryIO

def generate_gif(rng: BinaryIO, out: BinaryIO):
    header = b'GIF89a'
    width, height = struct.unpack('>HH', rng.read(4))
    lsd = struct.pack('>HHBHB', width, height, 0x87, 0x00, 0x00)
    color_table = rng.read(3)
    gce = b'\x21\xF9\x04\x00\x00\x00\x00\x00'
    image_descriptor = struct.pack('>BHHHHB', 0x2C, 0x0000, 0x0000, width, height, 0x00)
    lzw_minimum_code_size = 2
    data_sub_block = b'\x02\x04\x05'
    image_data = bytes([lzw_minimum_code_size]) + data_sub_block
    trailer = b'\x3B'
    out.write(header + lsd + color_table + gce + image_descriptor + image_data + trailer)
