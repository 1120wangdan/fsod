#3*3=9个特征向量和180个原型求相似度

import os
import cv2
import json
import torch 
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler

import torch
import torch.nn.functional as F

import torch.nn as nn
from einops import rearrange

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
        self.roi_pooler = ROIPooler(output_size=(3, 3), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        # self.roi_pooler1 = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
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
            
            #开始改位置
            # boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs] 
            count = 0
            boxes = []
            for x in inputs:
                last_two_values = x["instances"].gt_boxes.tensor.view(-1)[2:]
                min_value = torch.min(last_two_values)
                if min_value < 96:
                    count+=1
                    logger.info("缩放了{}次,缩放前bbox大小为:{}".format(count,x["instances"].gt_boxes.tensor.view(-1).tolist()))
                    ration = min_value / 96 
                    x["instances"].gt_boxes.tensor /= ration
                    img = cv2.resize(img,(int(img_h/ration), int(img_w/ration)))
                
                
                instance_boxes = x["instances"].gt_boxes.to(self.device)
                boxes.append(instance_boxes)
          #
          # extract roi featuresx
            #得到每张图片的特征   在小样本的标注下 一张图片只有一个目标 
            features = self.extract_roi_features(img, boxes)   #这是经过预处理的输入图像和输入图像上的坐标 
            all_features.append(features.cpu().data)  #cpu().data 方法，将特征数据移动到 CPU 上并获取其数据   
            

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)   
           

        # concat
        #将列表中的张量在维度0上进行连接得到一个张量
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
            # 将特征张量列表转换为张量

        # num_classes = len(features_dict)  # 类别的数量
        # num_prototype=10
        # in_channels=2048
        # prototypes = torch.zeros(num_classes, num_prototype, in_channels)  # 事先定义好的原型张量

        # index_to_label = {}  # 建立索引和类别标签之间的映射关系
        prototypes = {}
# 遍历字典中的每个类别
        for k, label in enumerate(features_dict):
            # 获取当前类别的特征张量列表
            features = features_dict[label]
            
            # 将特征张量列表转换为张量
            features = torch.cat(features, dim=0)
            
            # 调用 prototype_learning 方法，传递特征张量作为输入
            prototype = prototype_learning(features)
            
            # # 将原型存储到事先定义好的原型张量中
            # prototypes[k] = prototype   每一类的原型是10，2048
            prototypes[label] = prototype
            
            # # 建立索引和类别标签之间的映射关系
            # index_to_label[k] = label




            # for label in features_dict:
                 
            #      features_dict[label] = torch.cat(features_dict[label], dim=0)
        #     #unsqueeze(0) 扩展维度 方法的作用是在特征张量的第一个维度上增加一个维度，将其转换为形状为 (1, ...) 的张量。
       
        # #特征字典和原型字典用类别label关联
        # prototypes_dict = {}
        # for label in features_dict:
        #     features = torch.cat(features_dict[label], dim=0)   
        #     prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True) #每一类的像素平均作为原型
        #features_dict[label]中每个label作为键都存储着这一类对应的特征张量，遍历字典中的每一类，将每一类的特征作为输入传给prototype_learning方法，就可以返回该类对应的原型

        return prototypes 

            
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
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        

        box_features = self.roi_pooler([conv_feature], boxes) #移除指定维度
        a = box_features.size(0)
        box_features = box_features.view(a, 2048, 9)

        activation_vectors = torch.zeros(a,1000, 9)

        # 遍历box_features中的每个向量
        for i in range(box_features.size(2)):
            # 获取当前向量
            # vector = box_features[i]
            # print(i)
            vector = box_features[:, :, i]
         
            
            # 将向量通过全连接层
            output = self.imagenet_model.fc(vector)
            # print(output)
            activation_vectors[:, :, i]=output
            # 将输出添加到activation_vectors列表中

        activation_vectors_final = activation_vectors.reshape(a,1000, 3, 3)
        

        return activation_vectors_final   


    
     #PCB执行，求相似度、
     #extract_roi_features(img, boxes)的两个参数，一个来自原始图像，一个来自预测的候选框
     #dts是检测结果列表
 
    def execute_calibration(self, inputs, dts): 

        coco_annotation_file = '/root/workspace/DeFRCN/datasets/cocosplit/seed0/full_box_1shot_airplane_trainval.json'
        with open(coco_annotation_file, 'r') as f:
            coco_data = json.load(f)
        categories = coco_data['categories']
        category_dict = {category['id']: category['name'] for category in categories}   
        category_dict[0] = 'background'
        
        img = cv2.imread(inputs[0]['file_name'])   # BGR 格式的图像

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]  #HEAD给100个框

        features = self.extract_roi_features(img, boxes)

        f= open("output.txt","w")
        #i是框的索引吗 原论文也只和自己对应的类求相似度 
        for i in range(ileft, iright):
            f.write("\n")
            f.write(f"{i}th bbox")
            similarity_scores = []  
            similarity_labels = []  
           


            for k in range(9):
                cos_sim_list = []
                for label, prototypes_list in self.prototypes.items():
                    for j in range(9):
                        cos_sim = cosine_similarity(
                            np.reshape(l2_normalize_(features[i - ileft].reshape(1000,9)[:,k].cpu().data.numpy()), (1, -1)),
                            np.reshape(prototypes_list[j].cpu().data.numpy(), (1, -1))
                        )[0][0]
                        cos_sim_list.append(cos_sim)

                max_sim = max(cos_sim_list) 
                max_sim_index = cos_sim_list.index(max_sim) 
                max_sim_label = list(self.prototypes.keys())[max_sim_index // 9] 

                similarity_scores.append(max_sim)
                similarity_labels.append(max_sim_label)
            
            
            for key,value in zip(similarity_labels,similarity_scores):
                f.write(f"lable:{key},socre:{value}\n")
            #三层循环结束后得到了长度为k=9的相似度列表，
            # 统计每个类别出现的次数
            label_counts = {}
            for label in similarity_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1


            # 找出出现次数最多的类别
            max_count = max(label_counts.values())
            most_frequent_label = [label for label, count in label_counts.items() if count == max_count][0]

            # 筛选出现次数最多的相似度

            most_frequent_label_ = [most_frequent_label]  # 将 most_frequent_label 转换为列表
            most_frequent_scores = [score for score, label in zip(similarity_scores, similarity_labels) if label in most_frequent_label_]

            
            # 计算平均相似度
            average_similarity = sum(most_frequent_scores) / len(most_frequent_scores)

            # 将分类结果和分类得分记录下来


            cls = (dts[0]['instances'].pred_classes[i]).cpu().numpy().item()
            dts[0]['instances'].pred_classes[i] = most_frequent_label 
            score = dts[0]['instances'].scores[i]+0
            dts[0]['instances'].scores[i] = average_similarity


            box = dts[0]['instances'].pred_boxes[i].tensor.cpu().numpy().astype(int)

            # 提取左上角坐标和宽高
            x1, y1, width, height = box[0,0],box[0,1],box[0,2],box[0,3]

            # 绘制框和标注
            img_with_boxes = img.copy()   
            cv2.rectangle(img_with_boxes, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            text_line1 = f"bbox{i - ileft} - {category_dict[most_frequent_label]}:"
            text_line2 = f"{average_similarity:.4f}"
            cv2.putText(img_with_boxes, text_line1, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img_with_boxes, text_line2, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

           

            # 获取原图像文件名
            original_image_name = inputs[0]['file_name'].split('/')[-1]

            # 保存绘制后的图像，包含原图像名称和框编号
            output_image_name = f"{original_image_name}_bbox{i - ileft}.jpg"
            output_image_path = f"/root/workspace/DeFRCN/show/14_10/{output_image_name}"  # 替换为实际的保存路径
            cv2.imwrite(output_image_path, img_with_boxes)

            output_table = f"Image: {original_image_name}, Box Index: {i - ileft}\n"
            output_table += f"Predicted Label: {category_dict[most_frequent_label]}\n"
            output_table += f"Predicted Score: {average_similarity:.4f}\n"

            # 假设 cls 是类别的数字键值
            if cls in category_dict:
                output_table += f"Original Class: {category_dict[cls]}\n"
            else:
                output_table += f"Original Class: {cls}\n"

            # output_table += f"Original Class: {category_dict[cls]}\n"
            output_table += f"Original Score: {score:.4f}\n" 

            logger.info(output_table)

        f.close()
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



def l2_normalize_(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector
def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update
#函数的输入out是 N个像素点在K类上的相似度，相当于是代价矩阵，正则化参数，迭代次数


#_c是同一个类的roi_feature,b c h w
def prototype_learning(_c):
     in_channels = 1000
     num_prototype = 9
     protos = torch.zeros(num_prototype, in_channels)
     old_protos = torch.zeros(num_prototype, in_channels)
                                       
    #  dim_in=_c.size(1)
    #  proj_head  = nn.Sequential(
    #         nn.Conv2d(dim_in, dim_in, 1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(dim_in, in_channels, 1))
    #  feat_norm = nn.LayerNorm(in_channels)
    #  c = proj_head(_c)
     _c = rearrange(_c, 'b c h w -> (b h w) c')
    #  _c = feat_norm(_c)
     _c = l2_normalize(_c) 
     old_protos.data.copy_(l2_normalize(protos))
    
     scores = torch.einsum('nd,md->nm', _c, protos)
     q, indexs = distributed_sinkhorn(scores) 
     f = q.transpose(0, 1) @ _c
     f = F.normalize(f, p=2, dim=-1)
     new_value = momentum_update(old_value=old_protos, new_value=f,
                                            momentum=0.9, debug=False)
     protos = l2_normalize(new_value)
     return protos   #原型是9，2048

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    #初始化分配矩阵
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1] 
    K = L.shape[0]
   

    # make the matrix sums to 1 
    sum_L = torch.sum(L)
    L /= sum_L
    
    #以上两步就是非负和归一化，也就是转换为概率

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L , dim=1, keepdim=True)  #行和
        L /= K  #整个矩阵所有元素除以K

        L /= torch.sum(L, dim=0, keepdim=True)  #   没那么像的弄到另一个类别
        L /= B  #整个矩阵所有元素除以B

    L *= B  #B
    L = L.t() #转置
    # print(L)
    # a=torch.sum(L, dim=1, keepdim=True) 
    # print(a)
    # b=torch.sum(L, dim=0, keepdim=True) 
    # print(b)
    
    #返回最大值的列索引
    indexs = torch.argmax(L, dim=1)   #没有改变L本身
    # print(indexs)

    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)  #one-hot,就是一个掩码，指示每个像素对应的类，最终的分配结果
    # print(L)
    # a=torch.sum(L, dim=1, keepdim=True) 
    # print(a)
    # b=torch.sum(L, dim=0, keepdim=True) 
    # print(b)

    return L, indexs