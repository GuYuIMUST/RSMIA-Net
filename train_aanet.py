import pandas as pd

import dataIO
import transform
from utils import epoch_seg, scheduler, optim, metrics_inference, model_io, metrics, losses
import model_utils
#from models import AANet
from models import AANet6


from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed
import numpy as np
import torchvision.models
from tensorboardX import SummaryWriter
from optparse import OptionParser
import os
import functools
import random
import warnings
from inference import test_model

# torch.backends.cudnn.benchmark = True

GLOBAL_SEED = 0
# 设置全局种子为0

warnings.filterwarnings('ignore')


# 忽略警告信息

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义设置种子的函数，用于设置随机种子

def worker_init_fn(worker_id):
    # 获取当前worker的初始种子值，通过取模运算确保种子在合理范围内
    seed = torch.initial_seed() % 2 ** 32
    # 将当前worker的ID加上种子值，作为该worker的新种子
    set_seed(seed + worker_id)


# 定义一个工作初始化函数，根据 worker_id 来设置种子

if 'lustre' in os.getcwd():
    print('On Lustre')
    crop_size = [96, 96, 96]
else:
    print('On Local')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    crop_size = [96, 96, 96]
# 检查当前工作目录是否包含 'lustre'，并根据不同情况打印信息和设置 crop_size

DEVICE = 'cuda'
# 设置设备为 CUDA

image_root = './PEData/processed_itk/image'
label_root = './PEData/processed_itk/label'
vessel_root = './PEData/processed_itk/vessel'
save_model_dir = './save_models'
# 定义数据路径和保存模型文件夹路径


if __name__ == '__main__':
    # 如果当前脚本作为主程序运行，则执行以下代码

    parser = OptionParser()
    # 创建一个命令行参数解析器对象

    parser.add_option('--epochs',
                      dest='epochs',
                      default=500,
                      type='int',
                      help='number of epochs')
    # 添加一个命令行参数 --epochs，设置训练的 epoch 数，默认为 500

    parser.add_option('-b',
                      '--batch_size',
                      dest='batch_size',
                      default=4,
                      type='int',
                      help='batch size, CT in one batch')
    # 添加一个命令行参数 -b 或 --batch_size，设置批量大小，默认为 4

    parser.add_option('-s',
                      '--sample_num',
                      dest='sample_num',
                      default=2,
                      type='int',
                      help='sampled patch per CT. The actual image batch size is batch_size*sample_num')
    # 添加一个命令行参数 -s 或 --sample_num，设置每个 CT 的采样补丁数，默认为 2

    parser.add_option('-w',
                      '--num_workers',
                      dest='num_workers',
                      default=4,
                      type='int',
                      help='num_workers')
    # 添加一个命令行参数 -w 或 --num_workers，设置 num_workers，默认为 4

    parser.add_option('-l',
                      '--learn_rate',
                      dest='learn_rate',
                      default=1e-4,
                      type='float',
                      help='learn rate before warmup, will be *10 after warmup')
    # 添加一个命令行参数 -l 或 --learn_rate，设置学习率，默认为 1e-4

    parser.add_option('--hu_low',
                      dest='hu_low',
                      default=-100,
                      type='float',
                      help='hu_low')
    # 添加一个命令行参数 --hu_low，设置 hu_low，默认为 -100

    parser.add_option('--hu_high',
                      dest='hu_high',
                      default=500,
                      type='float',
                      help='hu_high')
    # 添加一个命令行参数 --hu_high，设置 hu_high，默认为 500

    parser.add_option('-o',
                      '--log-path',
                      type='str',
                      dest='log_path',
                      default='./log/',
                      help='log path')
    # 添加一个命令行参数 -o 或 --log-path，设置日志路径，默认为 './log/'

    parser.add_option('-u',
                      '--unique_name',
                      type='str',
                      dest='unique_name',
                      default='det3',
                      help='name of this experiment')
    # 添加一个命令行参数 -u 或 --unique_name，设置实验名称，默认为 'det2'

    parser.add_option('--seed',
                      type='int',
                      dest='seed',
                      default='1',
                      help='global random seed')
    # 添加一个命令行参数 --seed，设置全局随机种子，默认为 1

    parser.add_option('--valid',
                      type='int',
                      dest='valid',
                      default='1',
                      help='if valid')
    # 添加一个命令行参数 --valid，设置是否验证，默认为 1

    (options, args) = parser.parse_args()
    # 解析命令行参数并存储到 options 和 args 中

    set_seed(options.seed)
    # 设置随机种子为命令行参数中指定的值

    unique_name = options.unique_name
    # 获取实验名称

    model = AANet6.Net(n_filters=[64, 96, 128, 160], en_blocks=[2, 3, 3, 3], de_blocks=[1, 2, 2],
                      stem_filters=16, aspp_filters=32)
    # 创建一个模型对象 AANet.Net，并传入相关参数

    pretrain = True
    # 设置 pretrain 为 True

    if pretrain == True:
        state_dict = torch.load('./save_models/luna16_pretrain.pth')  # 加载预训练模型的参数
        model.load_state_dict(state_dict, strict=False)  # 将加载的参数应用到模型中
        model.init_head()  # 初始化模型的头部

    # model = nn.parallel.DataParallel(model)  # 将模型放入多个GPU上进行并行处理
    model.to(DEVICE)  # 将模型移动到指定的设备上（例如GPU）

    # 定义训练数据的数据增强操作
    transform_list_train = [
        transform.Pad(output_size=crop_size),  # 在图像周围填充以确保裁剪后的大小符合要求
        transform.RandomCrop(output_size=crop_size)  # 针对训练数据进行随机裁剪操作，增加数据多样性
    ]
    transform_list_valid = []  # 空的验证数据的数据增强操作

    # 创建训练和验证数据的数据增强操作组合
    train_transform = torchvision.transforms.Compose(transform_list_train)  # 将训练数据的增强操作组合起来
    valid_transform = torchvision.transforms.Compose(transform_list_valid)  # 将验证数据的增强操作组合起来

    # 定义裁剪工具
    crop_fn_train = dataIO.crop_itk.SegCrop2(  # 定义训练数据的裁剪工具
        crop_size=crop_size,  # 裁剪的大小
        tp_ratio=0.75,  # 正样本比例
        spacing=[1., 1., 1.],  # 间距
        rand_translation=[20, 20, 20],  # 随机平移范围
        rand_rotation=[30, 30, 30],  # 随机旋转范围
        rand_space=[0.8, 1.2],  # 随机空间变换范围
        rand_rotation90=[True, True, True],  # 是否进行90度随机旋转
        rand_flip=[True, True, True],  # 是否进行随机翻转
        rand_transpose=[False, False, False],  # 是否进行随机转置
        hu_low=options.hu_low,  # Hounsfield单位的最低值
        hu_high=options.hu_high,  # Hounsfield单位的最高值
        sample_num=options.sample_num,  # 样本数量
        obj_crop=True,  # 是否裁剪目标
        blank_side=0  # 空白边缘处理方式
    )
    crop_fn_valid = dataIO.crop_itk.SegCrop2(  # 定义验证数据的裁剪工具
        crop_size=crop_size,  # 裁剪的大小
        tp_ratio=0.75,  # 正样本比例
        spacing=[1., 1., 1.],  # 间距
        hu_low=options.hu_low,  # Hounsfield单位的最低值
        hu_high=options.hu_high,  # Hounsfield单位的最高值
        sample_num=options.sample_num,  # 样本数量
        obj_crop=False,  # 不裁剪目标
        blank_side=0  # 空白边缘处理方式
    )

    print('Dataset Prepare!')
    # 准备训练和验证数据集
    train_split = np.array(pd.read_csv('./PEData/train_split_cad.csv'))[:, 0].tolist()  # 读取训练数据的文件列表并转换为列表格式
    val_split = np.array(pd.read_csv('./PEData/test_split_cad.csv'))[:, 0].tolist()  # 读取验证数据的文件列表并转换为列表格式

    train_dataset = dataIO.dataset.DatasetITKV(  # 创建训练数据集对象
        image_root, label_root, vessel_root, train_split,  # 图像、标签、血管根目录及训练集文件列表
        crop_fn=crop_fn_train, transform_post=train_transform  # 裁剪函数和数据增强操作
    )
    valid_dataset = dataIO.dataset.DatasetITKV(  # 创建验证数据集对象
        image_root, label_root, vessel_root, val_split,  # 图像、标签、血管根目录及验证集文件列表
        crop_fn=crop_fn_valid, transform_post=valid_transform  # 裁剪函数和数据增强操作
    )

    print('Data Loader Prepare!')  # 打印提示信息
    # 创建训练和验证数据的数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=options.batch_size, shuffle=True,  # 训练数据加载器设置
        collate_fn=dataIO.dataset.collate_fn_dicts, worker_init_fn=worker_init_fn,  # 数据处理函数和工作初始化函数
        num_workers=options.num_workers, pin_memory=False, drop_last=True  # 工作进程数、是否使用固定内存、是否丢弃最后一个不完整的批次
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=options.batch_size, shuffle=False,  # 验证数据加载器设置
        collate_fn=dataIO.dataset.collate_fn_dicts, worker_init_fn=worker_init_fn,  # 数据处理函数和工作初始化函数
        num_workers=options.num_workers, pin_memory=False  # 工作进程数、是否使用固定内存
    )

    print('Loss, Metrics and Optimizer Prepare!')
    # 定义模型的评估指标和损失函数
    seg_metrics = [metrics.Fscore(threshold=0.4)]  # 定义分割模型评估指标，这里使用 F-score 指标

    loss = 0.5 * losses.TverskyLoss(square=True, batch_dice=True, alpha=0.5, beta=0.5) + \
           0.5 * losses.TverskyLoss(square=True, batch_dice=False, alpha=0.5, beta=0.5,
                                    drop_bg=True)  # 定义损失函数为 TverskyLoss 的组合

    seg_loss = model_utils.losses.SegLoss(  # 定义分割损失函数
        ['segs', 'segvs'], ['label', 'mask'], [1, 1], [[0], [0]],  # 分割模型输出名称、标签名称、权重、忽略索引
        [loss, loss], [0.75, 0.25]  # 损失函数及对应的权重
    )


    # seg_loss = model_utils.losses.SegLoss(  # 定义分割损失函数
    #     ['segs', 'segvs1', 'segvs2'],  # 分割模型输出名称
    #     ['label', 'mask', 'mask'],  # 对应的标签名称
    #     [1, 1, 1],  # 每个输出的权重
    #     [[0], [0], [0]],  # 忽略的索引
    #     [loss, loss, loss],  # 每个输出的损失函数
    #     [0.7, 0.15, 0.15]  # 每个损失函数的权重
    # )




    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(params=model.parameters(), lr=options.learn_rate, weight_decay=1e-4)  # AdamW 优化器
    scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.epochs,
                                                                  eta_min=1e-5)  # 余弦退火学习率调度器
    scheduler_warm = scheduler.GradualWarmupScheduler(  # 渐变热身学习率调度器
        optimizer, multiplier=10, total_epoch=4, after_scheduler=scheduler_reduce
    )
    grad_fn = functools.partial(optim.clip_grad_norm_, max_norm=100)  # 梯度裁剪函数

    print('Epoch Prepare!')  # 打印提示信息

    train_epoch = epoch_seg.TrainEpoch(  # 创建训练阶段的 Epoch 实例
        model=model,
        loss=seg_loss,
        seg_metrics=seg_metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        grad_fn=grad_fn
    )

    valid_epoch = epoch_seg.ValidEpoch(  # 创建验证阶段的 Epoch 实例
        model,
        loss=seg_loss,
        seg_metrics=seg_metrics,
        device=DEVICE,
        verbose=True,
    )

    print('Start Train!')
    set_seed(options.seed)  # 设置随机种子
    writer = SummaryWriter(options.log_path + unique_name)  # 创建用于记录日志的 SummaryWriter

    for i in range(0, options.epochs):  # 循环训练每个 epoch
        print('\nEpoch: {}'.format(i))  # 打印当前 epoch 数
        print('current lr:', optimizer.param_groups[0]['lr'])  # 打印当前学习率
        train_logs = train_epoch.run(train_loader)  # 在训练集上运行训练阶段

        model_io.Save(model.state_dict(), os.path.join(save_model_dir, unique_name + '_ms.pth'))  # 保存模型参数

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)  # 记录学习率变化
        for item in train_logs:  # 遍历训练日志项
            writer.add_scalar('Train/' + item, train_logs[item], i)  # 记录训练日志

        if options.valid:  # 如果需要验证
            valid_logs = valid_epoch.run(valid_loader)  # 在验证集上运行验证阶段
            for item in valid_logs:  # 遍历验证日志项
                writer.add_scalar('Valid/' + item, valid_logs[item], i)  # 记录验证日志

        scheduler_warm.step()  # 学习率调度器更新学习率

    "Test!"  # 打印提示信息
    model.eval()  # 将模型设置为评估模式
    test_model(model,
               './PEData/processed_itk/test_split_cad.csv',
               unique_name + '_sth60',
               seg_thresh=0.5,
               hu_low=options.hu_low, hu_high=options.hu_high)  # 在测试集上进行测试
