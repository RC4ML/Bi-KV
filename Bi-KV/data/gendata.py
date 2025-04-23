import pandas as pd


def gendata(dataset_name):
    # 加载 CSV 文件（假设文件名为 'data.csv'）
    # 如果文件不在当前工作目录，请提供完整路径
    csv_path = f'./{dataset_name}/{dataset_name}.csv'
    df = pd.read_csv(csv_path)
    df.columns = ['用户ID', '商品ID', '打分', '时间戳']

    # 步骤1：将时间戳按时间顺序转化为时间步
    df['时间步'] = df.groupby('用户ID')['时间戳'].rank(method='dense', ascending=True) - 1
    df['时间步'] = df['时间步'].astype(int)

    # 步骤2：统计每个用户的互动次数和时间步序列
    user_stats = df.groupby('用户ID').agg(
        互动次数=('时间步', 'count'),  # 统计每个用户的总互动次数
        时间步序列=('时间步', lambda x: list(x))  # 获取每个用户的时间步序列
    ).reset_index()

    import pickle
    pickle_path = f"./preprocessed/{dataset_name}_min_rating0-min_uc5-min_sc5/dataset.pkl"
    # 打开 .pkl 文件并加载数据
    with open(pickle_path, 'rb') as file:  # 'rb' 表示以二进制模式读取
        data = pickle.load(file)

    mapping_keys = set(data['umap'].keys())

    # 筛选出用户ID在映射表中的数据
    filtered_df = user_stats[user_stats['用户ID'].isin(mapping_keys)]
    filtered_df = filtered_df.reset_index(drop=True)

    # 将映射表中的数字作为“用户序号”加入到 filtered_df 中
    filtered_df['用户序号'] = filtered_df['用户ID'].map(data['umap'])
    filtered_df['时间步序列'] = filtered_df['时间步序列'].apply(sorted)

    filtered_df = filtered_df.sort_values(by='用户序号').reset_index(drop=True)

    # 使用 explode 方法将“时间步序列”列展开为多行
    expanded_df = filtered_df.explode('时间步序列')

    # 重命名列
    expanded_df = expanded_df.rename(columns={"时间步序列": "时间步"})

    # 转换“时间步”列为数值类型
    expanded_df['时间步'] = expanded_df['时间步'].astype(int)

    # 统计每个时间步中每个用户序号的访问次数
    time_step_counts = (
        expanded_df.groupby(['时间步', '用户序号']).size()
                .reset_index(name='访问次数')
    )

    # 构建映射表 map[时间步] -> list[(用户序号, 访问次数)]
    time_step_map = (
        time_step_counts.groupby('时间步')
                        .apply(lambda x: list(zip(x['用户序号'], x['访问次数'])))
                        .to_dict()
    )

    import json
    with open(f'./{dataset_name}/timestep_map.json', 'w') as f:
        json.dump(time_step_map, f, ensure_ascii=False, indent=4)
    print(f"已保存{dataset_name}/timestep_map.json")

    # 按商品ID分组并统计访问次数
    access_count_df = df.groupby("商品ID").size().reset_index(name="访问次数")
    access_count_df["商品ID"] = access_count_df["商品ID"].map(data['smap'])
    access_count_df.dropna(inplace=True)
    access_count_df["商品ID"] = access_count_df["商品ID"].astype(int)
    access_count_df.reset_index(drop=True, inplace=True)
    # 转换为字典格式 {商品ID: 访问次数}
    access_count_dict = dict(zip(access_count_df["商品ID"], access_count_df["访问次数"]))

    # 将字典保存为 JSON 文件
    with open(f'./{dataset_name}/item_access_count.json', "w", encoding="utf-8") as f:
        json.dump(access_count_dict, f, ensure_ascii=False, indent=4)
    print(f"已保存{dataset_name}/item_access_count.json")

if __name__ == "__main__":
    # dataset_name = 'games'
    for i in ['games','books','beauty','clothing']:
        print(f"正在处理数据集: {i}")
        gendata(i)