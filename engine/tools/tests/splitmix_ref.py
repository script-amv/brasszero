# SPDX-License-Identifier: Apache-2.0
"""Reference splitmix64 and its inverse, for tests that check seeds are not reversible."""
MASK = 0xFFFF_FFFF_FFFF_FFFF
_M1 = 0xBF58_476D_1CE4_E5B9
_M2 = 0x94D0_49BB_1331_11EB
_GAMMA = 0x9E37_79B9_7F4A_7C15


def splitmix64(x):
    z = (x + _GAMMA) & MASK
    z = ((z ^ (z >> 30)) * _M1) & MASK
    z = ((z ^ (z >> 27)) * _M2) & MASK
    return z ^ (z >> 31)


def _unxorshift(z, k):
    x = z
    for _ in range(64 // k + 1):
        x = z ^ (x >> k)
    return x


def unsplitmix64(z):
    z = _unxorshift(z, 31)
    z = (z * pow(_M2, -1, 1 << 64)) & MASK
    z = _unxorshift(z, 27)
    z = (z * pow(_M1, -1, 1 << 64)) & MASK
    z = _unxorshift(z, 30)
    return (z - _GAMMA) & MASK

