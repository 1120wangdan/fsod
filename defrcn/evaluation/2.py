import json
import os
from shutil import copyfile
from PIL import Image
import Augmentor

def augment_and_update_annotations(coco_data, output_path, selected_categories):
    for category in selected_categories:
        category_id = [cat['id'] for cat in coco_data['categories'] if cat['name'] == category][0]
        image_ids = [ann['image_id'] for ann in coco_data['annotations'] if ann['category_id'] == category_id]

        category_output_path = os.path.join(output_path, category)
        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)

        for image_id in image_ids:
            image_info = [img for img in coco_data['images'] if img['id'] == image_id][0]
            image_path = os.path.join(output_path, category, image_info['file_name'])
            output_image_path = os.path.join(category_output_path, image_info['file_name'])
            copyfile(image_path, output_image_path)

        # 数据增强
        p = Augmentor.Pipeline(category_output_path)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        
        augmented_images_path = os.path.join(output_path, 'augmented', category)
        p.sample(100, multi_threaded=False, output_directory=augmented_images_path)

        # 更新标注信息
        for annotation in coco_data['annotations']:
            if annotation['category_id'] == category_id:
                image_info = [img for img in coco_data['images'] if img['id'] == annotation['image_id']][0]
                image_path = os.path.join(category_output_path, image_info['file_name'])
                augmented_image_path = os.path.join(augmented_images_path, image_info['file_name'])

                # 更新边界框坐标
                annotation['bbox'] = update_bbox(annotation['bbox'], image_path, augmented_image_path)

                # 更新图像路径
                image_info['file_name'] = os.path.basename(augmented_image_path)

    return coco_data

def update_bbox(bbox, original_image_path, augmented_image_path):
    # 获取原始图像和增强后图像的尺寸差异
    original_image = Image.open(original_image_path)
    augmented_image = Image.open(augmented_image_path)
    width_ratio = augmented_image.width / original_image.width
    height_ratio = augmented_image.height / original_image.height

    # 更新边界框坐标
    updated_bbox = [
        bbox[0] * width_ratio,  # 更新x坐标
        bbox[1] * height_ratio,  # 更新y坐标
        bbox[2] * width_ratio,  # 更新宽度
        bbox[3] * height_ratio  # 更新高度
    ]

    return updated_bbox

# 设置COCO数据集路径和输出路径
coco_data_path = '/path/to/coco/data/'
output_path = '/path/to/augmented/coco/data/'

# 读取COCO数据集的标注文件
coco_annotation_file = os.path.join(coco_data_path, 'annotations', 'instances_train2017.json')
with open(coco_annotation_file, 'r') as f:
    coco_data = json.load(f)

# 设置要进行数据增强的类别
selected_categories = ['person', 'car']  # 你可以根据需要选择特定的类别

# 执行数据增强并更新标注信息
augmented_coco_data = augment_and_update_annotations(coco_data, output_path, selected_categories)

# 保存更新后的标注文件
output_annotation_file = os.path.join(output_path, 'augmented_instances_train2017.json')
with open(output_annotation_file, 'w') as f:
    json.dump(augmented_coco_data, f)

print("数据增强和标注信息更新完成。")
