import pandas as pd
from tianmao import TianmaoDataset  # 假设您的类保存在tianmao.py中

# 1. 创建数据集实例
class Args:
    min_rating = 1  # 最小评分(天猫数据没有评分，设为1)
    min_uc = 1      # 每个用户最少交互次数
    min_sc = 1      # 每个商品最少被交互次数

args = Args()
dataset = TianmaoDataset(args)

# 2. 检查原始数据加载
print("\n=== 检查原始数据 ===")
# 检查商品元数据
meta_dict = dataset.load_meta_dict()
print(f"加载到 {len(meta_dict)} 个商品的元数据")
print("示例商品:", list(meta_dict.items())[:3])

# 检查用户行为数据
ratings_df = dataset.load_ratings_df()
print(f"\n加载到 {len(ratings_df)} 条行为记录")
print("数据前5行:")
print(ratings_df.head())

# 3. 运行完整预处理
print("\n=== 运行完整预处理 ===")
full_dataset = dataset.load_dataset()

# 4. 检查预处理结果
print("\n=== 预处理结果 ===")
print(f"用户数量: {len(full_dataset['umap'])}")
print(f"商品数量: {len(full_dataset['smap'])}")
print(f"训练集用户数: {len(full_dataset['train'])}")
print(f"验证集用户数: {len(full_dataset['val'])}")
print(f"测试集用户数: {len(full_dataset['test'])}")

# 查看某个用户的划分情况
sample_user = 1
print(f"\n用户 {sample_user} 的数据划分:")
print(f"训练集商品数: {len(full_dataset['train'][sample_user])}")
print(f"验证集商品: {full_dataset['val'][sample_user]}")
print(f"测试集商品: {full_dataset['test'][sample_user]}")

# 查看商品元数据示例
sample_item = 1
print(f"\n商品 {sample_item} 的元数据:", full_dataset['meta'].get(sample_item, "无"))