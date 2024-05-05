import numpy as np
import os
import math
from collections import Counter
import tempfile
import cv2


def rgb_to_nv12(rgb_img):
    """
    convert rgb to nv12
    Args:
        rgb_img:

    Returns:

    """
    height, width, _ = rgb_img.shape
    uv_height = int(height / 2)
    uv_width = int(width / 2)
    yuv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV_I420)

    # extract Y,U,V
    yuv_planes = yuv_img.reshape(height * width + (uv_height * (uv_height + uv_width)))
    y_plane = yuv_planes[: height * width]
    u_plane = yuv_planes[height * width: height * (width + int(uv_width / 2))]
    v_plane = yuv_planes[height * (width + int(uv_width / 2)):]
    uv_plane = np.concatenate((u_plane[:, np.newaxis], v_plane[:, np.newaxis]), axis=1).flatten()

    # merge yuv to nv12
    nv12_img = np.concatenate((y_plane, uv_plane), axis=0, dtype=np.uint8)

    return nv12_img


def nv12_mat_to_yuv(img_nv12, height:int, width:int) -> np.array:
    """
    convert nv12 image to yuv
    Args:
        img_nv12:
        height:
        width:

    Returns:

    """
    f_y = img_nv12
    f_uv = img_nv12

    result = np.zeros((height, width, 3), dtype=np.uint8)
    size = int(3 / 2 * height * width)
    uv_start = height * width

    for j in range(0, height):
        for i in range(0, width):
            uv_index = uv_start + (width * math.floor(j / 2)) + math.floor(i / 2) * 2
            y = f_y[(j * width) + i]
            u = f_uv[int(uv_index)]
            v = f_uv[int(uv_index) + 1]

            result[j, i] = int(y), int(u), int(v)

    return result


def nv12_bin_to_yuv(nv12_file: str, height: int, width: int) -> np.array:
    """
    Given an NV12 file, return a 3 channel YUV image
    """
    f_y = open(nv12_file, "rb")
    f_uv = open(nv12_file, "rb")

    result = np.zeros((height, width, 3))

    size_of_file = os.path.getsize(nv12_file)
    size_of_frame = ((3.0 / 2.0) * height * width)
    number_of_frames = size_of_file / size_of_frame
    frame = 0
    frame_start = size_of_frame * frame
    uv_start = frame_start + (width * height)

    # lets get our y cursor ready
    f_y.seek(int(frame_start))
    for j in range(0, height):
        for i in range(0, width):
            # uv_index starts at the end of the yframe.  The UV is 1/2 height so we multiply it by j/2
            # We need to floor i/2 to get the start of the UV byte
            uv_index = uv_start + (width * math.floor(j / 2)) + (math.floor(i / 2)) * 2
            f_uv.seek(int(uv_index))

            y = ord(f_y.read(1))
            u = ord(f_uv.read(1))
            v = ord(f_uv.read(1))

            result[j, i] = int(y), int(u), int(v)

    return result


def yuv_to_rgb(yuv_data: np.array, target_img_format: str = 'RGB') -> np.array:
    """
    NOTE: Unable to use cv2 cvtColor func to convert YUV420p to RGB, so call this function for the time being

    Given a YUV image, convert it into an RGB image
    """
    pq_, bq_ = calculate_color_conversion_matrix('YUV', target_img_format)
    pq_ = pq_.reshape(3, 3)

    rgb_data = np.matmul(yuv_data, pq_.T) + bq_
    rgb_data = np.clip(rgb_data, 0, 255)

    return rgb_data


def calculate_general_prep_parameters(source_image_format, target_image_format, prep_mean, prep_scale, quant_scale):
    # 0. Checker
    assert source_image_format in ["RGB", "BGR", "YUV",
                                   "NV12"], f"source_image_format only supports ['RGB', 'BGR', 'YUV', 'NV12'], got {source_image_format}"

    assert target_image_format in ["RGB",
                                   "BGR"], f"target_image_format only supports ['RGB', 'BGR'], got {target_image_format}"

    assert isinstance(prep_mean, list) and len(prep_mean) == 3, f"prep_mean should be list type, and the length is 3"

    assert isinstance(prep_scale, list) and len(prep_scale) == 3, f"prep_scale should be list type, and the length is 3"

    for item in prep_scale:
        assert item % np.power(
            2.0,
            -12) == 0, "Make sure your scale values have an exact binary representation. In short, please make sure value / (2**-12) is an Integer."

    assert np.log2(quant_scale) == np.round(
        np.log2(quant_scale)), f"quant_scale is not a power_of_2 value, got {quant_scale}"

    # 1. Generate color conversion matrix
    ctype, ztype = source_image_format, target_image_format
    if source_image_format == "NV12":
        ctype = "YUV"

    color_p, color_b = calculate_color_conversion_matrix(ctype, ztype)

    # Hardware adder has total 20 bits.
    # Use left n bits (>=8) to store fixed point number,
    # while reminding bits for mantissa accuracy
    total_bit_number = 20

    num_shift = None

    inv_scale = 1 / np.array(quant_scale)
    prep_scale = np.array(prep_scale) * inv_scale

    # Image data range for RGB, YUV etc. Assume always use uint8 store image
    # data
    input_data_min = 0
    input_data_max = 255

    rmin = None
    rmax = None
    # calculate the data range after substract min and multiple scale
    for i in range(3):
        rmin = min(rmin, (input_data_min - prep_mean[i]) * prep_scale[i]
                   ) if rmin else (input_data_min - prep_mean[i]) * prep_scale[i]
        rmax = max(rmax, (input_data_max - prep_mean[i]) * prep_scale[i]
                   ) if rmax else (input_data_max - prep_mean[i]) * prep_scale[i]
    # use symmetric way (int8 quantization)
    rmax = max(abs(rmin), abs(rmax))
    data_range = 2 * rmax
    data_n_bits = np.ceil(np.log2(data_range))

    if data_n_bits <= 8.0:
        data_n_bits = 9
    num_shift = int(total_bit_number - data_n_bits)

    num_shift_scale = num_shift

    # compute and set PQ Matrix
    pq_arr = np.floor(
        np.diag(prep_scale).dot(np.array(color_p).reshape(3, 3)) * (1 <<
                                                                    num_shift_scale)).astype(int).flatten()

    # compute and set BQ Vector
    bq_arr = np.floor(np.diag(prep_scale).dot(
        np.array(color_b) - np.array(prep_mean)) * (1 << num_shift_scale)).astype(int).tolist()

    return pq_arr, bq_arr, num_shift


def prep_impl(input_data, pq_arr, bq_arr, num_shift):
    _, channel, height, width = input_data.shape
    result = np.zeros((1, channel, height, width))
    clamp_th_ = [[-128, 127], [-128, 127], [-128, 127]]
    for j in range(0, height):
        for i in range(0, width):
            y, u, v = input_data[0, :, j, i]
            r = np.int32(np.floor((pq_arr[0] * y + pq_arr[1] * u
                                   + pq_arr[2] * v + bq_arr[0]) * 2 ** -num_shift))
            g = np.int32(np.floor((pq_arr[3] * y + pq_arr[4] * u
                                   + pq_arr[5] * v + bq_arr[1]) * 2 ** -num_shift))
            b = np.int32(np.floor((pq_arr[6] * y + pq_arr[7] * u
                                   + pq_arr[8] * v + bq_arr[2]) * 2 ** -num_shift))

            r = np.clip(r, clamp_th_[0][0], clamp_th_[0][1])
            g = np.clip(g, clamp_th_[1][0], clamp_th_[1][1])
            b = np.clip(b, clamp_th_[2][0], clamp_th_[2][1])

            result[0, :, j, i] = int(r), int(g), int(b)

    return result


def calculate_color_conversion_matrix(in_type, out_type):
    # TODO: change ctype and ztype to in_type and out_type
    if in_type not in CTYPE_TO_CV2_FORMAT_MAPPING:
        raise RuntimeError(f"wrong ctype value {in_type}")
    if out_type not in CTYPE_TO_CV2_FORMAT_MAPPING:
        raise RuntimeError(f"wrong ztype value {out_type}")

    # base color_p and color_b
    color_p = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    color_b = np.array([0.0, 0.0, 0.0])

    # determine in/out category from mapping
    cat_in, cat_out = CTYPE_TO_CV2_FORMAT_MAPPING[in_type], CTYPE_TO_CV2_FORMAT_MAPPING[out_type]

    if Counter(cat_in) == Counter("YUV") and Counter(cat_out) == Counter("RGB"):
        # input of "YUV" category and output of "RGB" category
        color_p = np.array([[1.0, 0.0, 1.5748],
                            [1.0, -0.1873, -0.4681],
                            [1.0, 1.8556, 0.0]])
        color_b = np.array([-201.0744, 84.3912, -237.0168])

        # "cat_in to YUV" transformation
        # MUST get index from "YUV" default order!!!
        # yuv_index is column re-order
        yuv_index = ["YUV".index(cat_in[i]) for i in range(len("YUV"))]
        color_p = color_p[:, yuv_index]

        # "RGB" to cat_out transformation
        # MUST get index from "RGB" default order!!!
        # rgb_index is row re-order
        rgb_index = ["RGB".index(cat_out[i]) for i in range(len(cat_out))]
        color_p = color_p[rgb_index, :]
        color_b = color_b[rgb_index]

    elif Counter(cat_in) == Counter("RGB") and Counter(cat_out) == Counter("YUV"):
        # input of "RGB" category and output of "YUV" category
        color_p = np.array([[0.2126, 0.7152, 0.0722],
                            [-0.114572, -0.385428, 0.5],
                            [0.5, -0.454153, -0.045847]])
        color_b = np.array([0.1880, 127.9977, 127.9970])

        # "cat_in to RGB" transformation
        # MUST get index from "RGB" default order!!!
        # rgb_index is column re-order
        rgb_index = ["RGB".index(cat_in[i]) for i in range(len("RGB"))]
        color_p = color_p[:, rgb_index]
        # "YUV" to cat_out transformation
        # MUST get index from "YUV" default order!!!
        # yuv_index is row re-order
        yuv_index = ["YUV".index(cat_out[i]) for i in range(len(cat_out))]
        color_p = color_p[yuv_index, :]
        color_b = color_b[yuv_index]

    elif cat_out == 'GRAY':  # For gray scale images
        color_p[0, 0] = 1.0

    elif Counter(cat_in) == Counter(cat_out):
        # RGB -> BGR
        # input and output have the same category
        for i in range(len(cat_out)):
            color_p[i][cat_in.index(cat_out[i])] = 1.0
            # i = 0: color_p[0][2] = 1.0
            # i = 1: color_p[1][1] = 1.0
            # i = 2: color_p[2][0] = 1.0
            # 0, 0, 1
            # 0, 1, 0
            # 1, 0, 0
    # expect result of the form 1, N x N
    color_p = color_p.flatten()

    return color_p, color_b


CTYPE_TO_CV2_FORMAT_MAPPING = {
    "RGB": "RGB",
    "RGBD": "RGB",
    "RGBA": "RGB",
    "RBG": "RBG",
    "BRG": "BRG",
    "BGR": "BGR",
    "GBR": "GBR",
    "GRB": "GRB",
    "YUV": "YUV",
    "YUYV": "YUV",
    "YUVY": "YUV",
    "UYVY": "YUV",
    "NV12": "YUV",
    "NV21": "YUV",
    "YUV420P": "YUV",
    "YVU": "YVU",
    "VYU": "VYU",
    "VUY": "VUY",
    "UYV": "UYV",
    "UVY": "UVY",
    "GRAY": "GRAY",
    "MULTI_PD": "MULTI_PD"
}


def psnr_similarity(float_tensor, float_quantized_tensor):
    """
    PSNR computes the peak signal-to-noise ratio between two tensors.
    Refer to https://www.mathworks.com/help/vision/ref/psnr.html
    """
    if np.sum(np.abs(float_tensor)) == 0 or np.sum(np.abs(float_quantized_tensor)) == 0:
        float_tensor = np.add(float_tensor, 1e-5)
        float_quantized_tensor = np.add(float_quantized_tensor, 1e-5)
    mse = np.mean(np.power(float_tensor - float_quantized_tensor, 2))
    return 10 * np.log10(np.max(float_tensor) * np.max(float_tensor) / mse)


def mse_similarity(float_tensor, float_quantized_tensor):
    if np.sum(np.abs(float_tensor)) == 0 or np.sum(np.abs(float_quantized_tensor)) == 0:
        float_tensor = np.add(float_tensor, 1e-5)
        float_quantized_tensor = np.add(float_quantized_tensor, 1e-5)
    return np.mean(np.power(float_tensor - float_quantized_tensor, 2))


def rmse_similarity(float_tensor, float_quantized_tensor):
    if np.sum(np.abs(float_tensor)) == 0 or np.sum(np.abs(float_quantized_tensor)) == 0:
        float_tensor = np.add(float_tensor, 1e-5)
        float_quantized_tensor = np.add(float_quantized_tensor, 1e-5)
    return np.sqrt(np.mean(np.power(float_tensor - float_quantized_tensor, 2)))


if __name__ == "__main__":

    width = 640
    height = 640
    prep_mean = [128, 128, 128]
    prep_scale = [0.0078125, 0.0078125, 0.0078125]
    source_image_format = "NV12"
    target_image_format = "BGR"
    img_path = "../Datasets/bev_image_data/demo.jpg"
    nv12_path = "../Datasets/bev_image_data/demo.yuv"

    # rgb -> yuv
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    nv12_data = rgb_to_nv12(img_rgb)

    # nv12_mat -> yuv -> rgb
    yuv_data = nv12_mat_to_yuv(nv12_data, height, width)
    rgb_data = yuv_to_rgb(yuv_data, target_image_format)

    # nv12 -> bin
    with open(nv12_path, 'wb') as f:
        f.write(nv12_data.tobytes())

    # nv12 bin -> yuv
    yuv_data_1 = nv12_bin_to_yuv(nv12_path, height, width)
    rgb_data_1 = yuv_to_rgb(yuv_data_1, target_image_format)
    # rgb_data.astype(np.uint8).tofile("demo/test.rgb")

    cv2.imshow("img_0", rgb_data.astype(np.uint8))
    cv2.imshow("img_1", rgb_data_1.astype(np.uint8))
    cv2.waitKey()

    print("Done")
