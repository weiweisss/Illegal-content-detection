import pandas as pd
import numpy as np

def get_specific_label_samples(csv_path, label, n_samples, random_seed=42):
    """
    # 从CSV文件中随机抽取指定标签的样本
    # 参数:
    # csv_path: CSV文件路径
    # label: 需要提取的标签
    # n_samples: 需要抽取的样本数量
    # random_seed: 随机种子
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 筛选特定标签的数据
    label_df = df[df['bias_type'] == label]
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 如果请求的样本数量大于可用样本，则返回所有样本
    if n_samples > len(label_df):
        print(f"警告: 请求的样本数量({n_samples})大于可用样本数量({len(label_df)})")
        return label_df
    
    # 随机抽样
    sampled_df = label_df.sample(n=n_samples, random_state=random_seed)
    
    return sampled_df

# 使用示例
if __name__ == "__main__":
    csv_path = "merged.csv"
    specific_label = "rumor"  # 指定要提取的标签
    sample_size = 100   # 指定要提取的样本数量
    
    samples = get_specific_label_samples(csv_path, specific_label, sample_size)
    print(f"成功提取{len(samples)}个标签为{specific_label}的样本")
    
    # 保存提取的样本到新的CSV文件
    output_filename = f"label_{specific_label}_samples.csv"
    samples.to_csv(output_filename, index=False)
    print(f"样本已保存到文件: {output_filename}")
