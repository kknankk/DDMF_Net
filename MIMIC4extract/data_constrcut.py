#-----------------1--------------------
import pandas as pd

# # 定义文件路径
# admissions_file = "/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/admissions.csv"
# edstays_file = "/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ed/2.2/ed/edstays.csv"
# output_file = "/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/1.csv"

# # 加载 CSV 文件
# admissions_df = pd.read_csv(admissions_file)
# edstays_df = pd.read_csv(edstays_file)

# # 按 hadm_id 进行内连接
# merged_df = pd.merge(admissions_df, edstays_df, on="hadm_id", how="inner")

# # 提取所需的列
# selected_columns = [
#     "subject_id_x", "hadm_id", "admittime", "dischtime", "race_x",
#     "edregtime", "edouttime", "stay_id", "intime", "outtime", "gender", "race_y"
# ]
# merged_labels = merged_df[selected_columns]

# # 保存结果
# merged_labels.to_csv(output_file, index=False)

# print(f"合并并提取完成，结果已保存到 {output_file}")
#----------------1-------------------------


#------------------2----------------------------


# # # 读取 CSV 文件
# merged_labels = pd.read_csv('/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/1.csv')
# merged_labels.rename(columns={'subject_id_x': 'subject_id'}, inplace=True)

# merged_labels['admittime'] = pd.to_datetime(merged_labels['admittime'])
# merged_labels['dischtime'] = pd.to_datetime(merged_labels['dischtime'])
# merged_labels['edregtime'] = pd.to_datetime(merged_labels['edregtime'])
# merged_labels['endtime'] = merged_labels['edregtime'] + pd.Timedelta(hours=12)


# file_path = '/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
# cxr_metadata = pd.read_csv(file_path)
# cxr_metadata['StudyTime'] = cxr_metadata['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
# cxr_metadata['StudyDateTime'] = pd.to_datetime(cxr_metadata['StudyDate'].astype(str) + ' ' + cxr_metadata['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
# # print(cxr_metadata)
# cxr_metadata = cxr_metadata.rename(columns={'StudyDateTime': 'CXRStudyDateTime'})
# cxr_metadata1=cxr_metadata[['dicom_id', 'subject_id', 'ViewPosition', 'CXRStudyDateTime']]
# # # print(cxr_metadata1.head())

# # # 合并两个 DataFrame 按 subject_id 匹配



# merged_cxr = pd.merge(cxr_metadata1, merged_labels, on='subject_id', how='inner')

# # # 过滤出 StudyDateTime 在 admittime 和 dischtime 之间的行
# filtered_cxr = merged_cxr[(merged_cxr['CXRStudyDateTime'] >= merged_cxr['edregtime']) & 
#                         (merged_cxr['CXRStudyDateTime'] <= merged_cxr['endtime'])]
# filtered_cxr = filtered_cxr[filtered_cxr['ViewPosition'].isin(['AP', 'PA'])]


# # #---------ecg------------
# record_list_df = pd.read_csv('/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
# # # record_list_df = pd.read_csv(record_list)
# record_list_df['ecg_time'] = pd.to_datetime(record_list_df['ecg_time'])

# # # print(record_list_df.head())
# record_list_df1=record_list_df[['subject_id','ecg_time','path']]
# merged_df_ecg = pd.merge(record_list_df1, filtered_cxr, on='subject_id', how='inner')
# merged_df_ecg = merged_df_ecg[(merged_df_ecg['ecg_time'] >= merged_df_ecg['edregtime']) & 
#                         (merged_df_ecg['ecg_time'] <= merged_df_ecg['endtime'])]


# # # print(filtered_df)
# output_file_path = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/2.csv'
# merged_df_ecg.to_csv(output_file_path, index=False)

# print(f"Filtered DataFrame has been saved to {output_file_path}")

# #---------------------2----------------------

# #----------2.1----------------------



# input_file = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/2.csv'
# output_file = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/3.csv'


# columns_to_extract = [
#     'subject_id', 'hadm_id', 'edregtime', 'endtime', 'ecg_time',
#     'CXRStudyDateTime', 'path', 'dicom_id', 'ViewPosition',
#     'admittime', 'dischtime'
# ]

# data = pd.read_csv(input_file)
# filtered_data = data[columns_to_extract]


# # filtered_data.to_csv(output_file, index=False)
# # import pandas as pd


# # input_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2.csv'
# # output_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2_deduplicated.csv'


# # data = pd.read_csv(input_file)


# data_deduplicated = filtered_data.drop_duplicates()


# data_deduplicated.to_csv(output_file, index=False)

# #-----------2.1------------------

# #------------2.2----------------
# import pandas as pd

# # # 读取数据
# file_path = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/3.csv'
# data = pd.read_csv(file_path)


# filtered_rows = []


# for hadm_id, group in data.groupby('hadm_id'):
#     if len(group) == 1:
#        
#         filtered_rows.append(group.iloc[0])
#     else:
#        
#         unique_ecg_times = group['ecg_time'].drop_duplicates().tolist()
        
#         if len(unique_ecg_times) > 1:
#             
#             second_ecg_time = unique_ecg_times[1]  
            
#            
#             second_ecg_time_rows = group[group['ecg_time'] == second_ecg_time]
            
#             if len(second_ecg_time_rows) > 1:
#                 
#                 unique_cxr_times = second_ecg_time_rows['CXRStudyDateTime'].drop_duplicates().tolist()
#                 if len(unique_cxr_times) > 1:
#                     
#                     second_cxr_time = unique_cxr_times[1]
#                     final_row = second_ecg_time_rows[second_ecg_time_rows['CXRStudyDateTime'] == second_cxr_time].iloc[0]
#                 else:
#                    
#                     final_row = second_ecg_time_rows.iloc[0]
#             else:
#                
#                 final_row = second_ecg_time_rows.iloc[0]
            
#             filtered_rows.append(final_row)
#         else:
#            
#             continue


# result_df = pd.DataFrame(filtered_rows)
# result_df.to_csv('/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/4.csv', index=False)

# #---------------2.2--------------
# #---------------3----------------------
# import yaml

# icd_file_path = '/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv'
# df_diagnoses = pd.read_csv(icd_file_path)

# result_path = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/4.csv'
# result_df = pd.read_csv(result_path)


# with open('/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/category2icd_code9_10.yaml', 'r') as file:
#     category_data = yaml.safe_load(file)


# categories_to_include = {
#     key: value['codes'] for key, value in category_data.items() if value['use_in_benchmark']
# }
# # # print(f' categories_to_include {categories_to_include}')

# # import pandas as pd

# label_columns = [
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Pneumonia_or_COPD_or_OLRD_167'
# ]

# # Create binary columns for the specified categories
# for category, codes in categories_to_include.items():
#     df_diagnoses[category] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x) in codes else 0)
# # print(f'df_diagnoses {df_diagnoses.head()}')

# df_diagnoses = df_diagnoses.groupby('hadm_id', as_index=False).agg({

#     **{label: 'max' for label in label_columns}
# })
# # print(f'df_diagnoses {df_diagnoses.head()}')

# # #------------




# # new1_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_results.csv'
# save_path = '/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/5.csv'


# # filtered_results = pd.read_csv(new1_path)



# common_hadm_ids = result_df['hadm_id'].unique()
# df_diagnoses_filtered = df_diagnoses[df_diagnoses['hadm_id'].isin(common_hadm_ids)]


# columns_to_extract = [
#     'hadm_id',
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Pneumonia_or_COPD_or_OLRD_167'
# ]
# df_diagnoses_filtered = df_diagnoses_filtered[columns_to_extract]


# merged_results = pd.merge(
#     result_df,
#     df_diagnoses_filtered,
#     on='hadm_id',
#     how='inner'
# )


# merged_results.to_csv(save_path, index=False)



# #--------------3-----------------------

# #----------4.1 添加troponin T
# # import pandas as pd


a_df = pd.read_csv(save_path, parse_dates=['edregtime', 'endtime'])
b_df = pd.read_csv('/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/labevents.csv', parse_dates=['charttime'])


merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')

filtered_df = merged_df[
    (merged_df['charttime'] >= merged_df['edregtime']) &
    (merged_df['charttime'] <= merged_df['endtime'])
]


itemid_51002_df = filtered_df[filtered_df['itemid'] == 51003]


# unique_subject_ids_with_itemid_51002 = itemid_51002_df['subject_id'].nunique()
first_records_df = itemid_51002_df.groupby('subject_id').first().reset_index()

a_df = pd.merge(a_df, first_records_df[['subject_id', 'valuenum']], on='subject_id', how='left')


a_df = a_df.rename(columns={'valuenum': 'itemid_51003_valuenum'})


# print(f"a_df 中添加了 itemid_51003_valuenum 列：")
# print(a_df.head())
#------------
itemid_50908_df = filtered_df[filtered_df['itemid'] == 50908]
first_records_df_50908 = itemid_50908_df.groupby('subject_id').first().reset_index()
a_df = pd.merge(a_df, first_records_df_50908[['subject_id', 'valuenum']], on='subject_id', how='left')
a_df = a_df.rename(columns={'valuenum': 'itemid_50908_valuenum'})
# print(a_df.head())
itemid_50963_df = filtered_df[filtered_df['itemid'] == 50963]

first_records_df_50963 = itemid_50963_df.groupby('subject_id').first().reset_index()


a_df = pd.merge(a_df, first_records_df_50963[['subject_id', 'valuenum']], on='subject_id', how='left')
a_df = a_df.rename(columns={'valuenum': 'itemid_50963_valuenum'})
# a_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv', index=False)

# print("文件已成功保存到 /home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv")
# #----------------4.1


# # 文件路径
# # file_path = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/labevents.csv"


# # 读取 a.csv 和 b.csv
# # a_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv', parse_dates=['edregtime', 'endtime'])
# b_df = pd.read_csv('/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/labevents.csv', parse_dates=['charttime'])

# # 过滤 subject_id 匹配的行
# merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')

# # 筛选 charttime 在 admittime 和 dischtime 之间的记录
# filtered_df = merged_df[
#     (merged_df['charttime'] >= merged_df['edregtime']) &
#     (merged_df['charttime'] <= merged_df['endtime'])
# ]

# # 筛选 itemid 为 50963 的行
# itemid_50963_df = filtered_df[filtered_df['itemid'] == 50963]
# first_records_df_50963 = itemid_50963_df.groupby('subject_id').first().reset_index()

# # 检查是否有匹配结果
# if not first_records_df_50963.empty:
#     normalized_values = []
#         # normalized_values = []
#     subject_ids = []
#     for _, row in first_records_df_50963.iterrows():
#         ref_lower = row['ref_range_lower']
#         ref_upper = row['ref_range_upper']
#         value = row['valuenum']
        
#         # 检查参考范围是否有效
#         if pd.notnull(ref_lower) and pd.notnull(ref_upper) and ref_upper > ref_lower:
#             # 归一化计算
#             normalized_value = (value - ref_lower) / (ref_upper - ref_lower)
#             # 裁剪到 [0, 1]
#             # normalized_value = max(0, min(1, normalized_value))
#             normalized_values.append(normalized_value)
#             subject_ids.append(row['subject_id'])

#     # 将 normalized_values 添加到 a_df 的 itemid_50963_valuenum 列，覆盖原来的值
#     normalized_df = pd.DataFrame({'subject_id': subject_ids, 'itemid_50963_valuenum': normalized_values})

#     # 将归一化值合并到 a_df 中，覆盖原来的值
#     a_df = a_df.merge(normalized_df, on='subject_id', how='left', suffixes=('', '_new'))
#     a_df['itemid_50963_valuenum'] = a_df['itemid_50963_valuenum_new'].fillna(a_df['itemid_50963_valuenum'])
#     a_df.drop(columns=['itemid_50963_valuenum_new'], inplace=True)

# # 保存结果到文件（可选）
# # a_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv', index=False)
# #----------------5 add vital siganl

# # import pandas as pd


# # a_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_train.csv', parse_dates=['edregtime', 'endtime'])
# b_df = pd.read_csv('/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ed/2.2/ed/vitalsign.csv', parse_dates=['charttime'])


# merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')


# filtered_df = merged_df[
#     (merged_df['charttime'] >= merged_df['edregtime']) & 
#     (merged_df['charttime'] <= merged_df['endtime'])
# ]


# first_records_df = filtered_df.groupby('subject_id').first().reset_index()


# first_records_df = first_records_df[['subject_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']]

# a_df = pd.merge(a_df, first_records_df, on='subject_id', how='left')


# # print(f"a_df 中添加了相关的 vitalsign 列：")
# # print(a_df.head())


# a_df.to_csv('/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/MIMIC4extract/dataset_construct/2.csv', index=False)
