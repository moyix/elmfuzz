#!/usr/bin/env python3

import struct
from typing import BinaryIO

# Generates a random GIF file into `out` using the
# random number generator `rng` (/dev/urandom)
def generate_random_gif(
        rng: BinaryIO, out: BinaryIO
    ):
    header = b'GIF89a'
    datalen = int.from_bytes(
            rng.read(3), byteorder='big')
    data = rng.read(datalen)
    out.write(header + data)
