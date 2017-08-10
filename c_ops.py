import ctypes
import os
import sys
import numpy as np
import platform


if platform.system() == 'Linux':
    cur_path = sys.path[0]
    dll_path = os.path.join(cur_path, "tensorwolf", "kernel.so")
    c_kernel = ctypes.CDLL(dll_path)
else:
    cur_path = os.path.dirname(__file__)
    dll_path = os.path.join(cur_path, "c_kernel", "x64", "Release", "c_kernel.dll")
    c_kernel = ctypes.CDLL(dll_path)


def zero_padding_func(ori, up, down, left, right):
    ret = np.zeros([ori.shape[0], ori.shape[1] + up + down,
                    ori.shape[2] + left + right, ori.shape[3]])
    ret[:, up:up + ori.shape[1], left:left + ori.shape[2], :] = ori[:, :, :, :]
    return ret


def get_pointer(input):
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def correlate2d(input, filter, strides, padding):
    # setting shapes
    #assert input.dtype == np.float32
    #assert filter.dtype == np.float32
    batchs = input.shape[0]
    i_h = input.shape[1]
    i_w = input.shape[2]
    i_c = input.shape[3]
    f_h = filter.shape[0]
    f_w = filter.shape[1]
    assert i_c == filter.shape[2]
    o_c = filter.shape[3]
    # calc output
    if padding == 'SAME':
        output = np.zeros((batchs, i_h, i_w, o_c), dtype=np.float32)
        o_h = i_h
        o_w = i_w
        z_h = (i_h - 1) * strides[1] + f_h
        z_w = (i_w - 1) * strides[2] + f_w
        z = zero_padding_func(ori=input, up=(z_h - i_h) // 2, down=(z_h - i_h + 1) // 2,
                              left=(z_w - i_w) // 2, right=(z_w - i_w + 1) // 2)
    elif padding == 'VALID':
        o_h = (i_h - f_h + strides[1] - 1) // strides[1] + 1
        o_w = (i_w - f_w + strides[2] - 1) // strides[2] + 1
        output = np.zeros((batchs, o_h, o_w, o_c), dtype=np.float32)
        z_h = i_h
        z_w = i_w
        z = input
    else:
        raise NotImplementedError
    z = z.astype(np.float32)
    filter = filter.astype(np.float32)
    assert c_kernel.correlate2d(
        get_pointer(z),  # input's pointer as np.float32
        batchs,
        z_h,
        z_w,
        i_c,  # with its shape
        strides[1],
        strides[2],  # step
        get_pointer(filter),  # filter's pointer
        f_h,
        f_w,  # omit i_c here avoiding duplication
        o_c,  # with its shape
        get_pointer(output),
        o_h,
        o_w
    ) == 0
    # for some special reason(I don't know), there must be something between
    # return and c_kernel, otherwize BOOM...
    # print("np.sum = ", np.sum(output))
    return output


def conv2d_filter_gradient(input, gradient, ori_filter):
    # setting shapes
    batchs = input.shape[0]
    i_h = input.shape[1]
    i_w = input.shape[2]
    i_c = input.shape[3]
    f = gradient  # stupid me, it's just cor-relation. No rotation.
    # print("f_shape: ", f.shape)
    f_h = f.shape[1]
    f_w = f.shape[2]
    o_c = f.shape[3]
    o_h = ori_filter.shape[0]
    o_w = ori_filter.shape[1]
    z_h = i_h + o_h - 1
    z_w = i_w + o_w - 1
    z = zero_padding_func(ori=input, up=(z_h - i_h) // 2, down=(z_h - i_h + 1) // 2,
                          left=(z_w - i_w) // 2, right=(z_w - i_w + 1) // 2)
    output = np.zeros((o_h, o_w, i_c, o_c), dtype=np.float32)
    z = z.astype(np.float32)
    f = f.astype(np.float32)
    # print(z.shape, f.shape, output.shape)
    assert c_kernel.conv2d_filter_gradient(
        get_pointer(z),  # input's pointer as np.float32
        batchs,
        z_h,
        z_w,
        i_c,  # with its shape
        get_pointer(f),  # filter's pointer
        f_h,
        f_w,  # omit i_c here avoiding duplication
        o_c,  # with its shape
        get_pointer(output),
        o_h,
        o_w
    ) == 0
    return output


def max_pool_gradient(gradient, output, input, ksize, strides):
    assert ksize[1] == strides[1]
    assert ksize[2] == strides[2]
    g = gradient.astype(np.float32)
    input32 = input.astype(np.float32)
    assert c_kernel.max_pool_gradient(
        get_pointer(g),
        gradient.shape[0],  # batchs
        gradient.shape[1],  # g_h
        gradient.shape[2],  # g_w
        gradient.shape[3],  # ic
        get_pointer(output),
        ksize[1],  # h_step
        ksize[2],  # w_step
        get_pointer(input32),
        input.shape[1],  # z_h
        input.shape[2]  # z_w
    ) == 0
