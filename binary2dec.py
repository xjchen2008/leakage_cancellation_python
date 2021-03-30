import numpy as np


def bin2dec(num_bin):
    num_bin_str = np.binary_repr(num_bin, width = 16)
    if num_bin_str[0] == '0': # this is the most significant bit
        num_dec = num_bin
    else:
        num_dec = -1 * (np.invert(np.array(num_bin, dtype=np.uint16)) + 1)
    return num_dec


num_b = 1000000000001111

num_b = 1000000000001111#65534
num_d = bin2dec(num_b)
measured_angle = num_d * 0.018

print(num_d, measured_angle)


