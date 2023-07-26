def output_padding(stride, input_size, padding, kernel_size):
    output_padding = stride - (input_size - 1 + 2 * padding - (kernel_size - 1))
    return output_padding
