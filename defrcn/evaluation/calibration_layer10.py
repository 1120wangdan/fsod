import os
import cv2
import json
import torch 
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
import sys
sys.path.append("/root/workspace/DeFRCN/")


from sklearn.metrics.pairwise import cosine_similarity
from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101  

logger = logging.getLogger(__name__)

#论文中PCB模块是一个再1K上预训练过的分类网络

class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        #debug改成了1到3
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()
    
    #创建网络
    def build_model(self):
        #记录一条日志
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
       
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model
    
    #得到原型字典的函数,这个是要改的地方
    def build_prototypes(self):

        all_features, all_labels = [], []
        
        for index in range(len(self.dataloader.dataset)):   #循环执行每个样本  #从 self.dataloader.dataset 中获取的一个样本
            inputs = [self.dataloader.dataset[index]]   
            assert len(inputs) == 1
           
            # load support images and gt-boxes 
            img = cv2.imread(inputs[0]['file_name'])  # BGR 指定的图像文件
            img_h, img_w = img.shape[0], img.shape[1]  
            ratio = img_h / inputs[0]['instances'].image_size[0]  #原始图像和输入图像 求得尺度转换因子
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio #更新GT的框坐标，从原始图像的尺度转换到输入图像的尺度
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs] 

            # extract roi features
            #得到每张图片的特征向量
            features = self.extract_roi_features(img, boxes)   #这是经过预处理的输入图像和输入图像上的坐标
            all_features.append(features.cpu().data)  #cpu().data 方法，将特征数据移动到 CPU 上并获取其数据
           

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)   
            #假设所有样本的 gt_classes 字段都是一样的，因此只需要获取第一个样本的 gt_classes。 一张图里面的样本不一定是一种

        # concat
        all_features = torch.cat(all_features, dim=0)  #按照指定维度进行拼接
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == all_labels.shape[0]  #特征和标签匹配

        # calculate prototype
        
        #将特征按照类别进行分类得到分类的特征字典，每一类的用label作为键
        #特征的编号和类别label的编号
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            #如果不存在，则将其作为 键 添加到 features_dict 字典中，并将一个空列表作为对应的值。如果已经存在，则直接获取对应的值。
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))  
            #unsqueeze(0) 扩展维度 方法的作用是在特征张量的第一个维度上增加一个维度，将其转换为形状为 (1, ...) 的张量。
       
        #特征字典和原型字典用类别label关联
        prototypes_dict = {}
        for label in features_dict: 
            features = torch.cat(features_dict[label], dim=0)      #对每一类的特征进行拼接 
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True) #每一类的像素平均作为原型
            print(prototypes_dict)  

        return prototypes_dict
        
    #提取roi特征
    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]  
        images = ImageList.from_tensors(images, 0)

        #卷积特征-ROI特征-ROI特征向量
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1] #rgb
        # size: BxCxHxW
        box_features = self.roi_pooler([conv_feature], boxes)

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)  #移除指定维度

        activation_vectors = self.imagenet_model.fc(box_features)

        return box_features  

    #PCB执行，求相似度、
     #extract_roi_features(img, boxes)的两个参数，一个来自原始图像，一个来自预测的候选框
     #dts是检测结果列表
    def execute_calibration(self, inputs, dts):

        a=1
        img = cv2.imread(inputs[0]['file_name'])   # BGR 格式的图像
        #这里传入的有batch，
        
        #可能本来得分是降序排列的，个数的就是对应的索引
        #准确度大于1 的个数为0=不做矫正
        #right=100 准确度特别小的直接丢弃，
        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        # print(ileft)
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        # print(iright)
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]
        #列表-字典-属性

        features = self.extract_roi_features(img, boxes)
        #整个特征被拉成一个特征向量，估计效果也不会好  torch.Size([100, 1000])
        
        #i是框的索引吗 原论文也只和自己对应的类求相似度 
        for i in range(ileft, iright):   #遍历阈值内的框 i的作用
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            #原型和候选框的特征作度量 
            tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    #根据数据集名称返回一组需要排除的类别 ID 
    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


#在分布式环境下进行 all_gather 操作
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())] #返回的是当前分布式训练环境中的进程数量
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
