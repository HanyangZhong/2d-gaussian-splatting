# ++改过
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

# ++新增两个量
def loadCam(args, id, cam_info, resolution_scale, has_depth=False, has_normal=False):
    """
    加载摄像机信息，并根据需要加载深度图和法线图。

    :param args: 程序参数，包含配置选项
    :param id: 摄像机 ID
    :param cam_info: 摄像机信息，包含图像、位姿等
    :param resolution_scale: 分辨率缩放比例
    :param has_depth: 是否加载深度图
    :param has_normal: 是否加载法线图
    :return: Camera 对象，包含所有摄像机信息
    """
    orig_w, orig_h = cam_info.image.size

    # 处理分辨率调整逻辑
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # 处理图像和掩码
    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        # if has_depth:
        #     resized_depth_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.depth_image.split()[:3]], dim=0)
        #     loaded_depth_mask = PILtoTorch(cam_info.depth_image.split()[3], resolution)
        # if has_normal:
        #     resized_normal_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.normal_image.split()[:3]], dim=0)
        #     loaded_normal_mask = PILtoTorch(cam_info.normal_image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    resized_depth_image = None
    resized_normal_image = None

    if has_depth:
        resized_depth_image = PILtoTorch(cam_info.depth_image, resolution)
    if has_normal:
        resized_normal_image = PILtoTorch(cam_info.normal_image, resolution)

    # ++创建 Camera 对象
    camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,depth_image=resized_depth_image,normal_image=resized_normal_image,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    
    # ++如果需要加载深度图
    # if has_depth:
    #     depth_image = load_depth_image(cam_info.depth_image, resolution)
    #     camera.depth_image = depth_image

    # ++如果需要加载法线图
    # if has_normal:
    #     normal_map = load_normal_map(cam_info.normal_image, resolution)
    #     camera.normal_map = normal_map

    return camera

# ++加载深度图
def load_depth_image(depth_path, resolution):
    """
    从路径加载深度图并调整分辨率。

    :param depth_path: 深度图路径
    :param resolution: 目标分辨率
    :return: 调整分辨率后的深度图
    """
    from PIL import Image
    depth_image = Image.open(depth_path)
    depth_image = depth_image.resize(resolution, Image.BILINEAR)
    return PILtoTorch(depth_image, resolution)

# ++加载法线图
def load_normal_map(normal_map_path, resolution):
    """
    从路径加载法线图并调整分辨率。

    :param normal_map_path: 法线图路径
    :param resolution: 目标分辨率
    :return: 调整分辨率后的法线图
    """
    from PIL import Image
    normal_map = Image.open(normal_map_path)
    normal_map = normal_map.resize(resolution, Image.BILINEAR)
    return PILtoTorch(normal_map, resolution)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, has_depth=False, has_normal=False):
    """
    通过摄像机信息加载训练/测试摄像机列表，支持深度图和法线图的加载。
    
    :param cam_infos: 摄像机信息列表
    :param resolution_scale: 分辨率缩放比例
    :param args: 训练参数
    :param has_depth: 是否有深度图
    :param has_normal: 是否有法线图
    :return: 包含摄像机数据的列表
    """
    camera_list = []

    for id, cam_info in enumerate(cam_infos):
        camera = loadCam(args, id, cam_info, resolution_scale, has_depth, has_normal)
        camera_list.append(camera)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry