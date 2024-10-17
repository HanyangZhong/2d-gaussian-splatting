#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torchvision.transforms as T
from PIL import Image

# 保存 tensor 到 PNG 文件的函数
def save_tensor_as_image(tensor, path):
    """
    将 tensor 保存为 PNG 图像。
    """
    tensor = tensor.detach().cpu()  # 将 tensor 移到 CPU，并确保没有梯度
    tensor = torch.clamp(tensor, 0.0, 1.0)  # 将值限制在 [0, 1] 范围内
    transform = T.ToPILImage()
    image = transform(tensor)
    image.save(path)
    # print('saved in ',path)

import matplotlib.pyplot as plt
import numpy as np
# 保存 tensor 到 PNG 文件并叠加法向线条
def save_tensor_as_image_with_normals(image_tensor, normal_tensor, path, step=10, scale=10):
    """
    将 tensor 保存为 PNG 图像,并在图像上叠加法向线条。法向向量模长为1,并以点云为起点。
    
    :param image_tensor: 渲染的图像 tensor,shape 为 (3, H, W)
    :param normal_tensor: 法向 tensor,shape 为 (3, H, W)
    :param path: 保存的文件路径
    :param step: 控制法向线条的密度，每 step 个像素绘制一个法向线条
    :param scale: 法向线条的长度比例
    """
    image_tensor = image_tensor.detach().cpu()
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

    # 将渲染的图像 tensor 转换为 numpy 数组
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # 将法向 tensor 转换为 numpy 数组，并归一化法向模长
    normal_tensor = normal_tensor.detach().cpu()
    normal_np = normal_tensor.permute(1, 2, 0).numpy()
    norm_length = np.linalg.norm(normal_np, axis=-1, keepdims=True) + 1e-8  # 避免除以0
    normal_np = normal_np / norm_length  # 归一化法向

    # 创建一个 matplotlib figure
    plt.figure(figsize=(10, 10))

    # 显示渲染的图像
    plt.imshow(image_np)

    # 获取图像的形状
    height, width, _ = image_np.shape

    # 绘制法向线条
    for y in range(0, height, step):
        for x in range(0, width, step):
            # 获取法向方向（已经归一化）
            nx, ny, nz = normal_np[y, x]
            # 线条的起点是点云的位置
            start_point = (x, y)
            # 根据法向方向绘制线条，以点云为起点，法向为方向
            end_point = (x + int(nx * scale), y + int(ny * scale))
            # 绘制线条
            plt.arrow(x, y, nx * scale, ny * scale, color='red', head_width=1, head_length=1)

    # 关闭坐标轴
    plt.axis('off')

    # 保存图像
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 确保有路径
def ensure_directory_exists(path):
    """Ensure the directory for the given path exists."""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# 使用卷积核对法线进行平滑
import torch.nn.functional as F

def smooth_normals(normal_tensor, kernel_size=5):
    # 生成卷积核进行平滑
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=normal_tensor.device) / (kernel_size * kernel_size)

    # Apply convolution to each channel separately
    smoothed_channels = []
    for i in range(3):  # For each channel (x, y, z)
        channel = normal_tensor[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        smoothed_channel = F.conv2d(channel, kernel, padding=kernel_size//2)
        smoothed_channels.append(smoothed_channel.squeeze(0))  # Remove the extra dimensions

    # Stack the smoothed channels back together
    normal_smoothed = torch.cat(smoothed_channels, dim=0)
    return normal_smoothed

# 这是整个训练过程的核心函数。它的任务包括
# 1 初始化场景和模型
#       将数据集加载到高斯模型中，并创建场景对象。
# 2 迭代训练
#       通过循环（for iteration in range(first_iter, opt.iterations + 1)）逐步优化模型。在每次迭代中，随机选择一个视角渲染图像，然后计算损失并进行反向传播。
# 3 更新学习率和调整球谐函数（SH degree）
#       每经过一定的迭代次数，模型会提升球谐函数的阶数（SH degree），从而增加渲染的细节。
# 4 损失计算
#       主要损失函数是L1损失（像素间的绝对误差）和SSIM（结构相似性指标），并且包含正则化项（dist_loss 和 normal_loss）。
# 5 保存模型
#       在预定的迭代次数（如 saving_iterations）时，模型会保存当前的参数状态。
# 6 日志记录
#       使用TensorBoard记录每次迭代的损失值和训练时间。
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, path):
    first_iter = 0
    # step1 初始化场景
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_map_for_log = 0.0
    ema_normal_map_for_log = 0.0

    # ++深度图和法线图损失的计算
    # depth_loss, normal_image_loss = 0.0, 0.0
    depth_loss, normal_image_loss = torch.tensor(0.0), torch.tensor(0.0)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # step2 迭代训练  随机选择一个视角渲染图像    计算损失并进行反向传播
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # step3 更新学习参数
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 更新球谐函数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 随机选的视角 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # 提取里面的参数  图像  视角点  可视化滤波器   半径
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 取出原视角上的图片作为真值
        gt_image = viewpoint_cam.original_image.cuda()
        # Step4 计算 L1 loss  像素间的绝对误差
        Ll1 = l1_loss(image, gt_image)
        #         权重 反过来                          权重         SSIM（结构相似性指标）
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # ++如果场景中有深度图，则计算深度图损失
        if scene.has_depth:
            rendered_depth = render_pkg['surf_depth']  # 渲染得到的深度图
            gt_depth = viewpoint_cam.depth_image.cuda()  # 真实的深度图
            depth_loss = l1_loss(rendered_depth, gt_depth)  # 使用 L1 损失计算深度差异
            lambda_depth_loss = opt.lambda_depth_loss if iteration > 7000 else 0.0
            depth_loss = 0 * lambda_depth_loss * depth_loss
            # print('using Depth L1 as',depth_loss)

        # ++如果场景中有法线图，则计算法线图损失
        if scene.has_normal:
            rendered_normal = render_pkg['rend_normal']  # 渲染得到的法线图
            gt_normal = viewpoint_cam.normal_image.cuda()  # 真实的法线图

            # 对法线进行平滑处理
            smoothed_gt_normal = smooth_normals(gt_normal)
            smoothed_rendered_normal = smooth_normals(rendered_normal)

            normal_Ll1 = l1_loss(rendered_normal, gt_normal)

            # 使用余弦相似度计算法线对齐损失
            # cos_similarity = (smoothed_rendered_normal * smoothed_gt_normal).sum(dim=0)  # 渲染法线与真实法线的点积
            # normal_image_loss = 1.0 - cos_similarity.mean()  # 1 - 余弦相似度作为损失
            lambda_normal_image = opt.lambda_normal_image if iteration > 9000 else 0.0
            # 动态调整法线损失的权重
            # lambda_normal_image = min(0.05, 0.001 + (iteration / 20000) * 0.04)

            # normal_image_loss = 0* lambda_normal_image * normal_image_loss
            # normal的真值损失计算  先尝试与直接的图像进行迭代测试
            normal_image_loss = 1 * (lambda_normal_image * normal_Ll1 + lambda_normal_image * (1.0 - ssim(rendered_normal, gt_normal)))
            # print('using Normal L1 as',normal_image_loss)

            # 每10次迭代保存一次法线图
            if iteration % 500 == 0:
                # print('path ',scene.model_path)
                save_path_rendered = scene.model_path + f"/debug/rendered_normal_{iteration}.png"
                save_path_gt = scene.model_path + f"/debug/gt_normal_{iteration}.png"
                save_path_smooth_rendered = scene.model_path + f"/debug/smooth_rendered_normal_{iteration}.png"
                save_path_smooth_gt = scene.model_path + f"/debug/smooth_gt_normal_{iteration}.png"

                # 确保目录存在
                ensure_directory_exists(save_path_rendered)
                ensure_directory_exists(save_path_gt)

                # 保存渲染和真实的法线图
                save_tensor_as_image(rendered_normal * 0.5 + 0.5, save_path_rendered)  # 归一化到 [0, 1] 区间
                save_tensor_as_image(gt_normal * 0.5 + 0.5, save_path_gt)  # 归一化到 [0, 1] 区间
                # print(f"Saved rendered and GT normals for iteration {iteration}")

                # 保存渲染和真实的法线图，并叠加法向线条
                # save_tensor_as_image_with_normals(rendered_normal * 0.5 + 0.5, rendered_normal, save_path_rendered)  # 渲染的法向
                # save_tensor_as_image_with_normals(gt_normal * 0.5 + 0.5, gt_normal, save_path_gt)  # 真实的法向
                # print(f"Saved rendered and GT normals with normal lines for iteration {iteration}")


        # 下面都是属于正则化，没有真值，主要是约束
        # regularization
        # 法线一致性  权重
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        # 深度失真项 权重
        # lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # ++改 正则化的权重可以动态调节，或者直接关闭
        # lambda_normal = opt.lambda_normal if (iteration > 7000 and not scene.has_normal) else 0.0
        lambda_dist = opt.lambda_dist if (iteration > 3000 and not scene.has_depth) else 0.0

        # 深度失真项
        rend_dist = render_pkg["rend_dist"]
        # 渲染法线
        rend_normal  = render_pkg['rend_normal']
        # 表面法线
        surf_normal = render_pkg['surf_normal']
        # 法线误差 loss
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        # 深度失真
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        # 像素真值损失 + 深度失真损失  +  法线一致性损失
        # total_loss = loss + dist_loss + normal_loss
        
        # ++改 总损失：真值损失 + 正则化损失（如果适用）
        total_loss = loss + depth_loss + normal_image_loss + normal_loss + dist_loss

        total_loss.backward()

        iter_end.record()

        # Step5 更新
        with torch.no_grad():
            # Progress bar
            # ema平滑
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            # ++也加ema
            ema_depth_map_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_map_for_log
            ema_normal_map_for_log = 0.4 * normal_image_loss.item() + 0.6 * ema_normal_map_for_log

            # 每10次就处理一次显示
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "depth_map": f"{ema_depth_map_for_log:.{5}f}",
                    "normal_map": f"{ema_normal_map_for_log:.{5}f}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)

            # 迭代完就退出
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_map_loss', ema_depth_map_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_map_loss', ema_normal_map_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # 更新了每个高斯点的最大半径   通过调整高斯点的半径，模型能够更准确地表示那些在当前视角下显得更重要的高斯点（比如更靠近摄像机的点）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 会根据当前的 视角空间点 和 可见性过滤器 ，更新和记录密度化统计数据
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 到了致密化间隔了
                # 该过程会根据当前视角的点密度和梯度信息，动态增加或者修剪高斯点。密度化可以确保模型捕捉到更多的场景细节，而修剪则是为了去掉不必要的点，从而避免过多的计算负担。
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 密度化的梯度阈值 --- 这个值控制了哪些高斯点需要被加入密度化计算，基于它们的梯度大小
                    # 表示是否要根据点的不透明度来剔除高斯点 --- 如果某些点的透明度过高，它们可能会被剔除
                    # 表示场景中摄像机的范围 --- 可能会影响到密度化的点的选择
                    # 用于剔除点的大小阈值 --- 当迭代次数超过 opt.opacity_reset_interval 时，阈值为20，否则为 None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                # ++改 在迭代达到致密化间隔时，动态调整高斯点
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20
                #     # ++从深度图中计算梯度，检测几何变化区域
                #     if scene.has_depth:
                #         # ++要测试
                #         some_threshold = 5
                #         gt_depth = viewpoint_cam.depth_image.cuda()
                #         depth_gradients = torch.abs(torch.gradient(gt_depth))  # 计算深度图梯度
                #         high_gradient_mask = (depth_gradients > some_threshold)  # 根据阈值筛选高几何变化区域

                #         # ++将高几何变化区域加入到高斯点密度调整中
                #         gaussians.densify_and_prune(
                #             opt.densify_grad_threshold, 
                #             opt.opacity_cull, 
                #             scene.cameras_extent, 
                #             size_threshold, 
                #             high_gradient_mask
                #         )
                #     else:
                #         # ++ 如果没有深度图，则使用原来的密度化方式
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                # 在指定的迭代间隔（opt.opacity_reset_interval）后，或者在白色背景下初次密度化时，会对高斯点的不透明度进行重置
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # 更新模型参数
                gaussians.optimizer.step()
                # 梯度清零
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # 远程可视化连接
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    # 获取参数   
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        # 渲染包
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        # 图片
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        # 深度图的可视化和记录  本来就有
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        # 记录原始真值图像
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # 计算测试集 L1 和 PSNR 指标
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # 模型参数加载
    lp = ModelParams(parser)
    # 优化器参数
    op = OptimizationParams(parser)
    # 管道参数
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    # All done
    print("\nTraining complete.")