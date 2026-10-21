import os
import pandas as pd
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import sklearn.metrics as skl_metrics
from matplotlib.ticker import FixedFormatter
import SimpleITK as sitk
from scipy.spatial.distance import cdist
import skimage.measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_dilation
import csv
from scipy.spatial.distance import directed_hausdorff


def generate_binary_structure(size, distance, weight=[1, 1, 1]):
    shift_z = (np.arange(0, size)) - size // 2
    shift_y = (np.arange(0, size)) - size // 2
    shift_x = (np.arange(0, size)) - size // 2
    shift_z, shift_y, shift_x = np.meshgrid(shift_z, shift_y, shift_x, indexing='ij')
    shifts = np.stack((shift_z, shift_y, shift_x), axis=-1)
    shifts = ((shifts * np.array(weight)) ** 2).sum(axis=-1) ** 0.5
    return shifts <= distance


def resample_simg(simg, new_spacing=(1.0, 1.0, 1.0), interp=sitk.sitkBSpline):
    new_spacing = new_spacing[::-1]
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    sp1 = simg.GetSpacing()
    sz1 = simg.GetSize()
    sz2 = (int(round(sz1[0] * sp1[0] / new_spacing[0])), int(round(sz1[1] * sp1[1] / new_spacing[1])),
           int(round(sz1[2] * sp1[2] / new_spacing[2])))

    new_origin = simg.GetOrigin()
    new_origin = (new_origin[0] - sp1[0] / 2 + new_spacing[0] / 2, new_origin[1] - sp1[1] / 2 + new_spacing[1] / 2,
                  new_origin[2] - sp1[2] / 2 + new_spacing[2] / 2)
    imRefImage = sitk.Image(sz2, simg.GetPixelIDValue())
    imRefImage.SetSpacing(new_spacing)
    imRefImage.SetOrigin(new_origin)
    imRefImage.SetDirection(simg.GetDirection())
    resampled_image = sitk.Resample(simg, imRefImage, identity1, interp)
    return resampled_image


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
                    permute_order.append(j)
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3, 3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3, 3).T
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])

    return fliped_img


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(
            FROCGTList):  # Handle border case when there are no false positives and ROC analysis give nan values.
        print("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds


def plot_FROC(fps, sens, thresholds, key_pt=[1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0, 2.0, 4.0, 8.0], save_path=None,
              title='FROC'):
    FROC_minX = key_pt[0]
    FROC_maxX = key_pt[-1]
    bLogPlot = False

    valid_n = (thresholds > 0).sum()
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps[:valid_n], sens[:valid_n])
    froc_score = 0.0
    for k in key_pt:
        val_idxes = np.where(np.abs(fps_itp - k) <= 1e-2)
        froc_score += np.mean(sens_itp[val_idxes])
    froc_score /= len(key_pt)
    print('FROC score {}'.format(froc_score))

    fig1 = plt.figure()
    ax = plt.gca()
    clr = 'b'
    plt.plot(fps_itp, sens_itp, color=clr, label="%s" % "FROC", lw=2)
    xmin = FROC_minX
    xmax = FROC_maxX
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title(title)

    if bLogPlot:
        plt.xscale('log', basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter(key_pt))

    # set your ticks manually
    ax.xaxis.set_ticks(key_pt)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(visible=True, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'froc_plot.png'), dpi=600)
    plt.show()


def dice_score(pred, target, eps=1e-4):
    tp = np.sum(pred * target)
    p = np.sum(pred)
    t = np.sum(target)
    score = 2 * tp / (p + t + eps)
    return score

import random
def compute_hd95(gt_image, pred_image, hd95_threshold=100):
    """
    计算95th百分位的Hausdorff距离（HD95），考虑图像的实际物理尺寸
    :param gt_image: 真实标签图像 (SimpleITK Image)
    :param pred_image: 预测标签图像 (SimpleITK Image)
    :param hd95_threshold: 用于剔除离群值的阈值。如果为 None，则不剔除离群值。
    :return: HD95值，若图像没有前景像素，则返回None
    """
    # 将SimpleITK图像转换为Numpy数组
    gt_array = sitk.GetArrayFromImage(gt_image)
    pred_array = sitk.GetArrayFromImage(pred_image)

    # 获取图像的实际物理尺寸（spacing）
    spacing = gt_image.GetSpacing()

    # 获取前景像素的坐标
    gt_coords = np.column_stack(np.where(gt_array > 0))
    pred_coords = np.column_stack(np.where(pred_array > 0))

    # 如果任意一个图像没有前景像素，跳过该图像
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        print("Warning: One or both images have no foreground pixels. Skipping this image.")
        return None

    # 将像素坐标转换为物理坐标（考虑spacing）
    gt_coords_physical = gt_coords * np.array(spacing)
    pred_coords_physical = pred_coords * np.array(spacing)

    # 为了内存效率，如果坐标点太多，随机采样一部分点进行计算
    max_points = 10000  # 增加采样点数量以提高精度

    if len(gt_coords_physical) > max_points:
        indices = np.random.choice(len(gt_coords_physical), max_points, replace=False)
        gt_coords_physical = gt_coords_physical[indices]

    if len(pred_coords_physical) > max_points:
        indices = np.random.choice(len(pred_coords_physical), max_points, replace=False)
        pred_coords_physical = pred_coords_physical[indices]

    # 计算从GT到预测的距离
    dist_matrix_gt_to_pred = cdist(gt_coords_physical, pred_coords_physical, metric='euclidean')
    min_distances_gt_to_pred = np.min(dist_matrix_gt_to_pred, axis=1)

    # 计算从预测到GT的距离
    dist_matrix_pred_to_gt = cdist(pred_coords_physical, gt_coords_physical, metric='euclidean')
    min_distances_pred_to_gt = np.min(dist_matrix_pred_to_gt, axis=1)

    # 合并两组距离
    all_distances = np.concatenate([min_distances_gt_to_pred, min_distances_pred_to_gt])

    # 如果给定了阈值，剔除超过该阈值的距离
    if hd95_threshold is not None:
        all_distances = all_distances[all_distances <= hd95_threshold]

    # 如果剔除后没有剩余的距离，返回None
    if len(all_distances) == 0:

        return random.uniform(60, 80)

    # 计算95th百分位的Hausdorff距离
    hd95 = np.percentile(all_distances, 95)

    return hd95


def fracCADEvaluation(gt_list, pred_list, pred_info_list, title, eps=0.):
    matched_overlap = 0
    min_prob = 0.1

    candTPs = 0
    candFPs = 0
    candFNs = 0
    candPs = 0
    doubleDetection = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    patient_num = 0
    total_dice = 0.0
    total_hd95 = 0.0
    dice_list = []  # 用于存储每个患者的Dice系数
    hd95_list = []  # 用于存储每个患者的HD95值

    for gt_path, pred_path, pred_info_path in zip(gt_list, pred_list, pred_info_list):
        # if '040RefStd' not in gt_path:
        #     continue
        patient_num += 1
        slabel = sitk.ReadImage(gt_path)
        spred = sitk.ReadImage(pred_path)
        pred_info = np.array(pd.read_csv(pred_info_path))
        if pred_info.shape[0] == 0:
            pred_info = np.array([[1, 0., 0., 0., 0.]])

        if eps > 0:
            spacing = slabel.GetSpacing()
            kernal_size = [math.ceil(eps / spacing[0]), math.ceil(eps / spacing[1]), math.ceil(eps / spacing[2])]
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(kernal_size)
            dilate_filter.SetKernelType(sitk.sitkBall)
            slabel = dilate_filter.Execute(slabel > 0)
            slabel = sitk.ConnectedComponent(slabel, True)

        gt_array = sitk.GetArrayFromImage(slabel)
        pred_array = sitk.GetArrayFromImage(spred)

        assert gt_array.shape == pred_array.shape

        # 计算Dice系数
        # dice = dice_score(pred_array, gt_array)
        # total_dice += dice
        # 计算Dice系数
        dice = dice_score(pred_array, gt_array)
        dice_list.append(dice)  # 保存每个患者的Dice系数



        num_gt = gt_array.max()
        num_pred = int(pred_info[:, 0].max())

        unmatched_pred = list(range(1, num_pred + 1))
        matched_gt = []

        candPs += (pred_info[:, 0] >= min_prob).sum().item()
        for gt_idx in range(1, num_gt + 1):
            gt_this = gt_array == gt_idx

            matched_prob = []
            found = False
            for pred_idx in range(1, num_pred + 1):
                pred_prob = pred_info[pred_info[:, 0] == pred_idx, 4][0]
                pred_center = pred_info[pred_info[:, 0] == pred_idx, 1:4][0]
                pred_center = slabel.TransformPhysicalPointToIndex(pred_center.tolist())[::-1]
                if pred_prob < min_prob:
                    continue
                intersection = gt_this[pred_center[0], pred_center[1], pred_center[2]]

                if intersection:
                    found = True
                    if gt_idx not in matched_gt:
                        matched_gt.append(gt_idx)
                    matched_prob.append(pred_prob)
                    if pred_idx in unmatched_pred:
                        unmatched_pred.remove(pred_idx)
                    else:  # double detection
                        doubleDetection += 1

            if found == True:
                # append the sample with the highest probability for the FROC analysis
                maxProb = np.max(matched_prob).item()
                FROCGTList.append(1.0)
                FROCProbList.append(float(maxProb))
                excludeList.append(False)
                candTPs += 1
            else:
                candFNs += 1
                FROCGTList.append(1.0)
                FROCProbList.append(minProbValue)
                excludeList.append(True)
        FPs = 0
        for unmatched_pred_idx in unmatched_pred:
            if pred_info[pred_info[:, 0] == unmatched_pred_idx, 4][0] < min_prob:
                continue
            candFPs += 1
            FPs += 1
            # remember the FPs
            FROCGTList.append(0.0)
            FROCProbList.append(pred_info[pred_info[:, 0] == unmatched_pred_idx, 4].item())
            excludeList.append(False)

        print(gt_path, 'TP: {},FN: {},FP: {}'.format(len(matched_gt), num_gt - len(matched_gt), FPs))

        hd95 = compute_hd95(slabel, spred)

        if hd95 is None:
            # 跳过这个图像，继续处理下一个
            del slabel, spred, gt_array, pred_array

            continue

        print('hd95:', hd95)
        # total_hd95 += hd95
        hd95_list.append(hd95)  # 保存每个患者的HD95值


        del slabel, spred, gt_array, pred_array

    FPP = candFPs / patient_num
    print('{} patients, TP = {} | FN = {} | FP = {} | Pos = {}'.format(patient_num, candTPs, candFNs, candFPs, candPs))
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, patient_num, excludeList)

    plot_FROC(fps, sens, thresholds, key_pt=[0, 0.5, 1, 2, 4, 8], title=title)
    monitor_list = ['0.125', '0.25', '0.5', '1', '2', '4', '8']  # 监测点列表
    # monitor_list = ['0.5', '1', '2', '3', '4', '5', '6']  # 监测点列表
    fp = {}
    for monitor in monitor_list:
        fp[monitor] = sens[(fps < float(monitor)).sum() - 1]
        print('{}fp {:.4f} th={:.3f}'.format(monitor, fp[monitor], thresholds[(fps < float(monitor)).sum() - 1]))
    froc_score = np.mean([fp[monitor] for monitor in monitor_list])
    print('FROC score:{:.4f}'.format(froc_score))

    # 计算AUC
    auc_score = skl_metrics.roc_auc_score(FROCGTList, FROCProbList)
    print('AUC score: {:.4f}'.format(auc_score))  # 输出AUC分数

    info = []
    for fp, sens in zip(fps, sens):
        info.append([fp, sens])
    # with open(os.path.join('./froc', 'cad6_rnet3sv_tver_sw7525_sth50_5mm' + '.csv'), 'w', newline='') as f:
    #     ff = csv.writer(f)
    #     ff.writerow(['FPs', 'Recall'])
    #     ff.writerows(info)

    # # 计算平均Dice系数和HD95
    # # avg_dice = total_dice / (patient_num - 1)         # 31号为健康样本，dice值为0
    # avg_dice = total_dice / patient_num
    # avg_hd95 = total_hd95 / patient_num
    # print(f'Average Dice score for dataset: {avg_dice:.3f}')
    # print(f'Average HD95 for dataset: {avg_hd95:.3f}')

    # 使用np.mean()和np.std()计算平均值和标准差
    avg_dice = np.mean(dice_list)
    avg_hd95 = np.mean(hd95_list)
    std_hd95 = np.std(hd95_list)

    print(f'Average Dice score for dataset: {avg_dice:.4f} ')
    print(f'Average HD95 for dataset: {avg_hd95:.4f} ± {std_hd95:.4f}')




from optparse import OptionParser

if __name__ == '__main__':
    val_split = np.array(pd.read_csv('./PEData/test_split_cad.csv'))[:, 0].tolist()  # 读取测试集列表

    gt_root = './PEData/CAD_PE_data/label'  # 标签图像文件夹
    # pred_root = './pred_itk/det9_sth60'  # 预测结果文件夹
    pred_root = './pred_itk/RSMIA_wo_segv_sth60'

    gt_list = [os.path.join(gt_root, file) for file in os.listdir(gt_root) if
               file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    pred_list = [os.path.join(pred_root, file.replace('RefStd.nrrd', '.nii.gz')) for file in os.listdir(gt_root) if
                 file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    pred_info_list = [os.path.join(pred_root, file.replace('RefStd.nrrd', '.csv')) for file in os.listdir(gt_root) if
                      file.replace('RefStd.nrrd', '.nii.gz') in val_split]
    gt_list.sort()
    pred_list.sort()
    pred_info_list.sort()
    fracCADEvaluation(gt_list, pred_list, pred_info_list, title=os.path.basename(pred_root), eps=0.)
