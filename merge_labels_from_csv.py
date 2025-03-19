import pandas as pd
import random

# 读取CSV文件
df = pd.read_csv('merged.csv')

# 确保选择偶数个样本
number = 40
sample_size = number if number % 2 == 0 else len(df) - 1
selected_indices = random.sample(range(len(df)), sample_size)

# 将选中的数据分成两组进行配对
pairs = []
bias_pairs = []
for i in range(0, len(selected_indices), 2):
    idx1 = selected_indices[i]
    idx2 = selected_indices[i + 1]
    
    # 合并文本内容
    combined_text = str(df.iloc[idx1]['context']) + ' ' + str(df.iloc[idx2]['context'])
    
    # 合并bias_type
    bias_types = set()
    if pd.notna(df.iloc[idx1]['bias_type']):
        bias_types.update(df.iloc[idx1]['bias_type'].split(','))
    if pd.notna(df.iloc[idx2]['bias_type']):
        bias_types.update(df.iloc[idx2]['bias_type'].split(','))
    
    combined_bias = ','.join(sorted(bias_types)) if bias_types else ''
    
    pairs.append(combined_text)
    bias_pairs.append(combined_bias)

# 创建新的DataFrame并保存
new_df = pd.DataFrame({
    'text': pairs,
    'bias_type': bias_pairs
})

# 保存到新的CSV文件
new_df.to_csv('merged_paired.csv', index=False)
print(f"处理完成，共生成 {len(pairs)} 对组合数据")
