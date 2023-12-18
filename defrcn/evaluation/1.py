# import numpy as np

# def bayesian_fusion_probability_vector(prob_vector_A, prob_vector_B, weight_A, weight_B):
#     # 贝叶斯融合公式
#     fused_probability_vector = (weight_A * np.array(prob_vector_A) + weight_B * np.array(prob_vector_B)) / (weight_A + weight_B)
#     return fused_probability_vector

# # 两个模型的分类概率向量（示例中有3个类别）
# prob_vector_model_A = [0.2, 0.5, 0.3]
# prob_vector_model_B = [0.4, 0.3, 0.3]

# # 两个模型的权重
# weight_model_A = 0.6
# weight_model_B = 0.4

# # 调用贝叶斯融合函数
# result_probability_vector = bayesian_fusion_probability_vector(prob_vector_model_A, prob_vector_model_B, weight_model_A, weight_model_B)

# print("贝叶斯融合后的分类概率向量:", result_probability_vector)

import numpy as np

def naive_bayesian_fusion(predictions_model_A, predictions_model_B, weight_model_A, weight_model_B):
    # 计算每个模型的概率分布（假设已经归一化）
    prob_model_A = predictions_model_A / np.sum(predictions_model_A)
    prob_model_B = predictions_model_B / np.sum(predictions_model_B)    

    # 使用朴素贝叶斯融合的权重计算融合后的概率分布
    fused_prob = weight_model_A * prob_model_A + weight_model_B * prob_model_B

    return fused_prob

# 两个模型的预测概率分布（示例中有3个类别）
predictions_model_A = np.array([0.2, 0.5, 0.3])
predictions_model_B = np.array([0.4, 0.3, 0.3])

# 两个模型的权重
weight_model_A = 0.6
weight_model_B = 0.4

# 调用朴素贝叶斯融合函数
result_prob = naive_bayesian_fusion(predictions_model_A, predictions_model_B, weight_model_A, weight_model_B)

print("贝叶斯融合后的概率分布:", result_prob)


#两个1都可以不除 