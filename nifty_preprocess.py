import SimpleITK as sitk
from scipy.ndimage import binary_closing, generate_binary_structure
from scipy import ndimage
import pandas as pd
import csv
import time
from skimage.measure import regionprops, label
from skimage.morphology import disk
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.itkprocess import skeletonSegment

# 定义一个函数，用于将图像重采样为指定的像素间距和插值方法
import SimpleITK as sitk
import numpy as np


# 定义一个函数，用于将图像进行重采样
def resample_simg(simg, new_spacing=(1.0, 1.0, 1.0), interp=sitk.sitkBSpline):
    new_spacing = new_spacing[::-1]  # 将像素间距倒置，因为SimpleITK中的像素间距顺序是z,y,x
    identity1 = sitk.Transform(3, sitk.sitkIdentity)  # 创建一个3D单位变换矩阵

    sp1 = simg.GetSpacing()  # 获取原始像素间距
    sz1 = simg.GetSize()  # 获取原始图像大小
    sz2 = (int(round(sz1[0] * sp1[0] / new_spacing[0])), int(round(sz1[1] * sp1[1] / new_spacing[1])),
           int(round(sz1[2] * sp1[2] / new_spacing[2])))  # 计算新图像大小

    new_origin = simg.GetOrigin()
    new_origin = (new_origin[0] - sp1[0] / 2 + new_spacing[0] / 2, new_origin[1] - sp1[1] / 2 + new_spacing[1] / 2,
                  new_origin[2] - sp1[2] / 2 + new_spacing[2] / 2)  # 计算新的原点位置
    imRefImage = sitk.Image(sz2, simg.GetPixelIDValue())  # 新建一个指定尺寸和像素类型的图像
    imRefImage.SetSpacing(new_spacing)
    imRefImage.SetOrigin(new_origin)
    imRefImage.SetDirection(simg.GetDirection())
    resampled_image = sitk.Resample(simg, imRefImage, identity1, interp)  # 对原始图像进行重采样
    return resampled_image


# 定义一个函数，用于将图像重新定向为指定的方向
def redirection_simg(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3, 3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3, 3).T

    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)  # 找到目标方向和当前方向之间的对应关系
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)  # 调整图像轴的顺序
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3, 3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3, 3).T
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)  # 计算需要翻转的轴
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])  # 翻转图像的轴

    return fliped_img


ct_roots = ['./PEData/CAD_PE_data/image']
label_roots = ['./PEData/CAD_PE_data/label']
mask_roots = ['./PEData/CAD_PE_data/vessel']
vessel_roots = ['./PEData/CAD_PE_data/vessel']

save_ct_root = './PEData/processed_itk/image'
save_label_root = './PEData/processed_itk/label'
save_vessel_root = './PEData/processed_itk/vessel'


# 遍历ct_roots, label_roots, mask_roots, vessel_roots中对应位置的目录
for ct_root, label_root, mask_root, vessel_root in zip(ct_roots, label_roots, mask_roots, vessel_roots):
    # 获取当前CT图像目录下的所有文件
    files = os.listdir(ct_root)

    # 遍历当前CT图像目录下的所有文件
    for file in files:
        # 构建各个文件的完整路径
        ct_path = os.path.join(ct_root, file)  # CT图像路径
        label_path = os.path.join(label_root, file.replace('.nrrd', 'RefStd.nrrd'))  # 标注图像路径
        mask_path = os.path.join(mask_root, file.replace('.nrrd', '.nii.gz').replace('.nii.gz', '_lungmask.nii.gz'))  # 肺部Mask图像路径
        vessel_path = os.path.join(vessel_root, file.replace('.nrrd', '.nii.gz'))  # 血管Mask图像路径

        file_name = os.path.basename(ct_path).replace('.nrrd', '.nii.gz')  # 获取文件名（去掉后缀.nrrd并更改为.nii.gz）

        # 加载数据
        simg = sitk.ReadImage(ct_path)  # 读取CT图像
        slabel = sitk.ReadImage(label_path)  # 读取标注图像
        smask = sitk.ReadImage(mask_path)  # 读取肺部Mask图像
        svessel = sitk.ReadImage(vessel_path)  # 读取血管Mask图像

        # 打印像素间距
        print(simg.GetSpacing())

        # redirection
        if np.abs(
                np.array(simg.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() > 1:
            print(file_name, 'redirection')
            simg = redirection_simg(simg)  # 重新定向CT图像
            slabel = redirection_simg(slabel)  # 重新定向标注图像
            smask = redirection_simg(smask)  # 重新定向肺部Mask图像
            svessel = redirection_simg(svessel)  # 重新定向血管Mask图像

        np_image = sitk.GetArrayFromImage(simg)  # 将CT图像转换为numpy数组
        np_label = sitk.GetArrayFromImage(slabel).astype(np.uint8)  # 将标注图像转换为numpy数组并转换为uint8类型
        np_mask = sitk.GetArrayFromImage(smask).astype(np.uint8)  # 将肺部Mask图像转换为numpy数组并转换为uint8类型
        np_vessel = sitk.GetArrayFromImage(svessel).astype(np.uint8)  # 将血管Mask图像转换为numpy数组并转换为uint8类型

        assert np_mask.shape == np_image.shape  # 确保肺部Mask图像与CT图像尺寸相同
        assert np_image.shape == np_label.shape  # 确保CT图像与标注图像尺寸相同

        # lung area
        roi_bbox = regionprops(np_mask.astype(int))[0].bbox  # 获取肺部Mask图像的bounding box
        np_image = np_image[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]  # 裁剪CT图像
        np_label = np_label[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]  # 裁剪标注图像
        np_vessel = np_vessel[roi_bbox[0]:roi_bbox[3], roi_bbox[1]:roi_bbox[4], roi_bbox[2]:roi_bbox[5]]  # 裁剪血管Mask图像

        # in case embolism is not contained by vessel
        np_vessel[np_label > 0] = 1  # 将标注图像中的肺栓塞区域所在的位置设置为血管Mask图像中的位置

        itk_image = sitk.GetImageFromArray(np_image)  # 将numpy数组转换为ITK图像
        itk_image.SetSpacing(slabel.GetSpacing())  # 设置像素间距
        itk_label = sitk.GetImageFromArray(np_label)  # 将numpy数组转换为ITK图像
        itk_label.SetSpacing(slabel.GetSpacing())  # 设置像素间距
        itk_vessel = sitk.GetImageFromArray(np_vessel)  # 将numpy数组转换为ITK图像
        itk_vessel.SetSpacing(slabel.GetSpacing())  # 设置像素间距

        sitk.WriteImage(itk_image, os.path.join(save_ct_root, file_name))  # 保存CT图像
        sitk.WriteImage(itk_label, os.path.join(save_label_root, file_name))  # 保存标注图像
        sitk.WriteImage(itk_vessel, os.path.join(save_vessel_root, file_name))  # 保存血管Mask图像
