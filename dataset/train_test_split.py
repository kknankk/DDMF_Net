

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 文件路径
input_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv'
val_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
test_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'

# 加载数据
data = pd.read_csv(input_csv_path)

# 定义疾病列
diseases = [
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]

# 填充缺失值
data[diseases] = data[diseases].fillna(0)

# 创建标签列，确保每个标签的组合都保持比例
combined_labels = data[diseases].apply(lambda x: ''.join(map(str, x)), axis=1)

# 将 `combined_labels` 添加到 `data` 中作为新列
data['combined_labels'] = combined_labels

# 按 subject_id 划分数据集
subject_ids = data['subject_id'].unique()

# 为每个 subject_id 生成一个标签（疾病组合）
subject_labels = data.groupby('subject_id')['combined_labels'].first()

# 定义一个函数，对每个标签进行分层划分
def stratified_split(data, label_col, test_size=0.5):
    # 对每个标签的正负样本分别分层
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()

    for label in data[label_col].unique():
        label_data = data[data[label_col] == label]
        
        # 按比例划分每个标签的数据
        label_train, label_val = train_test_split(label_data, test_size=test_size, random_state=0)
        
        # 将划分的训练集和验证集加入最终的数据集
        train_data = pd.concat([train_data, label_train], axis=0)
        val_data = pd.concat([val_data, label_val], axis=0)

    return train_data, val_data

# 使用自定义分层函数
train_data, val_data = stratified_split(data, label_col='combined_labels')

# 检查训练集和验证集每个标签的正负样本比例
for disease in diseases:
    print(f"Train {disease} negative/positive: {train_data[disease].value_counts(normalize=True)}")
    print(f"Val {disease} negative/positive: {val_data[disease].value_counts(normalize=True)}")

# 创建文件夹
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)

# # 保存数据到相应的文件夹
# train_data.to_csv(os.path.join( 'with_nonan_label_PA_train.csv'), index=False)
# val_data.to_csv(os.path.join('with_nonan_label_PA_val.csv'), index=False)

train_data.to_csv(os.path.join(val_folder, 'with_nonan_label_PA_val.csv'), index=False)
val_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)
# test_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)


print("Data has been split and saved successfully.")
