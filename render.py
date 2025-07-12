from typing import NamedTuple
import torch.nn as nn
import torch
import os
import json
import math
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from matplotlib import pyplot as plt
from PIL import Image
import time
from vanilla_gaussian_splatting import rasterize_gaussians, mark_visible

from pynvml import *
from contextlib import contextmanager

@contextmanager
def measure_gpu_memory(tag=""):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info_before = nvmlDeviceGetMemoryInfo(handle).used
    yield
    info_after = nvmlDeviceGetMemoryInfo(handle).used
    nvmlShutdown()
    diff = (info_after - info_before) / 1024 / 1024
    print(f"[{tag}] GPU memory delta: {diff:.2f} MB")



class Camera:
    def __init__(self, camera_json, device):
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        self.width = camera_json['width'] // 2
        self.height = camera_json['height'] // 2
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx']
        self.focal_y = camera_json['fy']
        self.zFar = 1000.0
        self.zNear = 0.01
        self.device = device
        self.view_mat = self.get_viewmatrix()
        self.proj_mat = self.get_projmatrix()
        self.tan_fovx = self.get_tanfovx()
        self.tan_fovy = self.get_tanfovy()

    def get_tanfovx(self):
        return self.width / (2 * self.focal_x)
    
    def get_tanfovy(self):
        return self.height / (2 * self.focal_y)
    
    def get_viewmatrix(self):
        c2w = torch.eye(4)
        c2w[:3, :3] = self.rotation
        c2w[:3, 3] = self.position
        w2c = torch.inverse(c2w)
        self.cam_pos = c2w[:3, 3].to(self.device)
        return w2c.transpose(0, 1).to(self.device)
    
    def get_projmatrix(self):
        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P
        fovx = 2 * np.arctan(self.get_tanfovx())
        fovy = 2 * np.arctan(self.get_tanfovy())
        proj_mat = getProjectionMatrix(self.zNear, self.zFar, fovx, fovy).transpose(0, 1).to(self.device)
        full_mat = (self.get_viewmatrix().unsqueeze(0).bmm(proj_mat.unsqueeze(0))).squeeze(0)
        return full_mat



class Rasterizer(nn.Module):
    def __init__(self, gaussain_path, sh_degree, device="cuda"):
        super().__init__()
        self.device = device
        self.active_sh_degree = sh_degree
        self.load_ply(gaussain_path, sh_degree)
        

    def load_ply(self, path, sh_degree=3):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        features_rest = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()

        self.means3D = torch.tensor(xyz, dtype=torch.float, device=self.device)
        self.shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
        self.opacities = torch.tensor(opacities, dtype=torch.float, device=self.device)
        self.scales = torch.tensor(scales, dtype=torch.float, device=self.device)
        self.rotations = torch.tensor(rots, dtype=torch.float, device=self.device)
        self.active_sh_degree = sh_degree

        self.scales = torch.exp(self.scales)
        self.opacities = torch.sigmoid(self.opacities)
        self.rotations = torch.nn.functional.normalize(self.rotations)
        print(f"Loaded {self.opacities.shape[0]} gaussians")

    def forward(self, camera, background):
        colors_precomp = torch.Tensor([])
        cov3Ds_precomp = torch.Tensor([])

        # torch.save({
        #     "raster_settings.bg":background, 
        #     "means3D" :self.means3D,
        #     "colors_precomp" :colors_precomp,
        #     "opacities" :self.opacities,
        #     "scales" :self.scales,
        #     "rotations" :self.rotations,
        #     "raster_settings.scale_modifier" :1,
        #     "cov3Ds_precomp" :cov3Ds_precomp,
        #     "raster_settings.viewmatrix" :camera.view_mat,
        #     "raster_settings.projmatrix" :camera.proj_mat,
        #     "raster_settings.tanfovx" :camera.tan_fovx,
        #     "raster_settings.tanfovy" :camera.tan_fovy,
        #     "raster_settings.image_height" :camera.height,
        #     "raster_settings.image_width" :camera.width,
        #     "sh" :self.shs,
        #     "raster_settings.sh_degree" :self.active_sh_degree,
        #     "raster_settings.campos" :camera.cam_pos,
        #     "raster_settings.prefiltered" :False,
        #     "raster_settings.debug" :False
        # }, r"D:\code\VanillaGS\output\test/vanilla_gs1.pth")
        

        # Invoke C++/CUDA rasterization routine
        rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer = rasterize_gaussians(
            background,
            self.means3D,
            colors_precomp,
            self.opacities,
            self.scales, 
            self.rotations,
            1,
            cov3Ds_precomp,
            camera.view_mat,
            camera.proj_mat,
            camera.tan_fovx,
            camera.tan_fovy,
            camera.height, 
            camera.width,
            self.shs,
            self.active_sh_degree,
            camera.cam_pos,
            False,
            False)
        return rendered, out_color


def render_scene(gaussian_path, camera_path, save_path=None, sh_degree=3, test_idx=None):
    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)  # black
    
    with open(camera_path, 'r') as camera_file:
        cameras_json = json.loads(camera_file.read())

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    rasterizer = Rasterizer(gaussian_path, sh_degree, device)

    for i in range(10):
        rasterizer.forward(Camera(cameras_json[0], device), bg_color)
    if test_idx is not None:
        cameras_json = [cameras_json[test_idx]]

    times = []
    test_count = 1
    progress_bar = tqdm(cameras_json)
    for i,camera_json in enumerate(progress_bar):
        camera = Camera(camera_json, device)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(test_count):
            num_rendered, image = rasterizer.forward(camera, bg_color)
        torch.cuda.synchronize()
        t1 = time.time()

        times.append((t1-t0)*1000 / test_count)
        progress_bar.set_description(f"FPS = {1/(t1-t0) :2f}, time = {(t1-t0)*1000:.2f} ms, num_rendered = {num_rendered}")

        if save_path is not None:
            image_path = os.path.join(save_path, "%s.jpg" % camera.img_name.split('/')[-1])
            image = image.permute(1,2,0).cpu().numpy()
            image = (image*255).astype(np.uint8)
            Image.fromarray(image).save(image_path.replace('.ppm', '.jpg'))
    print(f"Mean time cost = {np.mean(times):.2f} ms, Max time cost = {np.max(times):.2f} ms")
    plt.plot(times, label="Render time of VanillaGS")
    plt.xlabel("Frame")
    plt.ylabel("Time cost (ms)")
    plt.title("Render time of VanillaGS")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    scenes = {
        "garden": {
            "gaussian_path": r"/mnt/e/Dataset/GaussianSplattingModels/garden/point_cloud/iteration_30000/point_cloud.ply",
            "camera_path": r"/mnt/e/Dataset/GaussianSplattingModels/garden/cameras.json",
            "save_path": r"output/garden/vanilla"
        },
        "mc_aerial_c36": {
            "gaussian_path": r"D:\data\mc_aerial_c36\point_cloud\iteration_30000\point_cloud.ply",
            "camera_path": r"D:\data\mc_aerial_c36\cameras.json",
            "save_path": r"output/mc_aerial_c36/vanilla"
        },
        "DP1258_dji_and_yy": {
            "gaussian_path": r"D:\data\DP1258_dji_and_yy_merge\point_cloud\iteration_7000\point_cloud.ply",
            "camera_path": r"D:\data\DP1258_dji_and_yy_merge\cameras.json",
            "save_path": r"output/DP1258_dji_and_yy/vanilla"
        },
        "DP1139": {
            "gaussian_path": r"D:\data\DP1139\high_res_merged\point_cloud\iteration_30000\point_cloud.ply",
            "camera_path": r"D:\data\DP1139\high_res_merged\cameras.json",
            "save_path": r"output/DP1139/vanilla"
        },
        "37009-47004": {
            "gaussian_path": r"D:\data\37009-47004\point_cloud\iteration_30000\point_cloud.ply",
            "camera_path": r"D:\data\37009-47004\cameras.json",
            "save_path": r"output/37009-47004/vanilla",
        },
    }
    
    scene = "garden"
    gaussian_path = scenes[scene]["gaussian_path"]
    camera_path = scenes[scene]["camera_path"]
    save_path = scenes[scene]["save_path"]
    with measure_gpu_memory("vanilla rendering"):
        render_scene(gaussian_path, camera_path, save_path=None, sh_degree=3, test_idx=None)