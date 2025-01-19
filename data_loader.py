import os
import sys
import collections
import struct
import numpy as np
import cv2

from PIL import Image as PILImage

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
CameraInfo = collections.namedtuple(
    "CameraInfo", ["uid", "R", "T", "image_path", "depth_path", "image_name"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    def qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
             2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
             2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
             1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
             2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
             2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
             1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        uid = intr.id

        # 需要转换到TSDF Fusion需要的格式
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        T = -R.dot(T)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        depth_path = image_path.replace("/images/", "/depths/").replace(".jpg", ".png")
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, image_path=image_path, depth_path=depth_path, image_name=image_name)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readGaustudioColmap(path):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    images_dir = os.path.join(path, "images")
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=images_dir)

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    intrinsic_path = os.path.join(path, "intrinsic.txt")
    if os.path.exists(intrinsic_path):
        with open(intrinsic_path, 'r') as f:
            intrinsic_matrix = np.array([list(map(float, line.split())) for line in f.readlines()])
    else:
        fx_sum = 0
        fy_sum = 0
        cx_sum = 0
        cy_sum = 0
        count = 0

        for id, camera in cam_intrinsics.items():
            fx, fy, cx, cy = camera.params
            fx_sum += fx
            fy_sum += fy
            cx_sum += cx
            cy_sum += cy
            count += 1

        avg_fx = fx_sum / count
        avg_fy = fy_sum / count
        avg_cx = cx_sum / count
        avg_cy = cy_sum / count

        intrinsic_matrix = np.array([
            [avg_fx, 0, avg_cx],
            [0, avg_fy, avg_cy],
            [0, 0, 1]
        ])

        with open(intrinsic_path, 'w') as f:
            for row in intrinsic_matrix:
                f.write(" ".join(map(str, row)) + "\n")

    return cam_infos, intrinsic_matrix


def read_depth_meter(depth_path):
    # Read depth_max from image
    depth_img = PILImage.open(depth_path)
    png_info = depth_img.info
    if "depth_max" in png_info:
        depth_max = float(png_info["depth_max"])
    else:
        raise ValueError("depth_max not found in the PNG metadata!")

    # Read depth image
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(float)
    depth_im = (depth_im / 65535.0) * depth_max
    depth_im[depth_im >= 8.0] = 0  # set invalid depth to 0 (specific to iphone dataset)

    return depth_im
