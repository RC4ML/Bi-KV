import random
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import pickle
import os

class TianmaoDataset:
    def __init__(self, min_uc=1, min_sc=1):
        self.min_uc = min_uc
        self.min_sc = min_sc
        self.raw_data_folder = Path('/data/zzm/Bi-KV/data/Tianmao')
        self.processed_folder = Path('/data/zzm/Bi-KV/data/preprocessed/Tianmao')
        os.makedirs(self.processed_folder, exist_ok=True)
    
    def process(self):
        print("Loading raw data...")
        df = self._load_actions()
        meta = self._load_meta()
        
        print(f"原始用户行为数据记录数: {len(df)}")
        print(f"原始元数据商品数: {len(meta)}")
        
        clean_meta = {k.replace('i', ''): v for k, v in meta.items()}
        df['clean_sid'] = df['sid'].str.replace('i', '')
        df = df[df['clean_sid'].isin(clean_meta.keys())]
        df = df.drop(columns=['clean_sid'])
        
        print(f"过滤后行为数据记录数: {len(df)}")
        
        if len(df) == 0:
            print("\n调试信息:")
            print("行为数据中的唯一商品ID示例:", df['sid'].unique()[:10])
            print("元数据中的唯一商品ID示例:", list(meta.keys())[:10])
            print("去除'i'前缀后的元数据ID示例:", list(clean_meta.keys())[:10])
            raise ValueError("过滤后数据为空，请检查商品ID匹配情况")
        
        df = self._filter_interactions(df)
        print(f"最终有效行为数据记录数: {len(df)}")
        
        user_history = self._build_user_history(df)
        
        dataset = {
            'user_history': user_history,
            'meta': clean_meta,
            'statistics': {
                'num_users': len(df['uid'].unique()),
                'num_items': len({k for k in clean_meta if k in df['sid'].str.replace('i', '').unique()}),
                'total_interactions': len(df)
            }
        }
        
        output_path = self.processed_folder / 'tianmao_dataset.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset processed and saved to {output_path}")
        return dataset
    
    def _load_actions(self):
        file_path = self.raw_data_folder / '(sample)sam_tianchi_2014002_rec_tmall_log.csv'
        
        try:
            df = pd.read_csv(file_path, sep=',', header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=',', header=0, encoding='gbk')
        
        df = df.rename(columns={
            'item_id': 'sid',
            'user_id': 'uid',
            'vtime': 'timestamp'
        }).astype({'sid': str, 'uid': str})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        
        return df[['uid', 'sid', 'action', 'timestamp']]
    
    def _load_meta(self):
        file_path = self.raw_data_folder / '(sample)sam_tianchi_2014001_rec_tmall_product.csv'
        
        try:
            df = pd.read_csv(file_path, sep=',', header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=',', header=0, encoding='gbk')
        
        df = df.rename(columns={'item_id': 'sid'}).astype({'sid': str})
        
        df['title'] = df['title'].map(lambda x: None if str(x).lower() == 'null' else str(x).strip())
        df = df.dropna(subset=['title'])
        
        return df.set_index('sid')['title'].to_dict()
    
    def _filter_interactions(self, df):
        print("Filtering low-frequency users and items...")
        
        while True:
            start_len = len(df)
            user_counts = df['uid'].value_counts()
            good_users = user_counts[user_counts >= self.min_uc].index
            df = df[df['uid'].isin(good_users)]
            
            item_counts = df['sid'].value_counts()
            good_items = item_counts[item_counts >= self.min_sc].index
            df = df[df['sid'].isin(good_items)]
            
            if len(df) == start_len:
                break
                
        return df
    
    def _build_user_history(self, df):
        print("Building user history...")
        
        df = df.astype({'uid': str, 'sid': str})
        
        user_history = {}
        for uid, group in tqdm(df.groupby('uid'), desc="Processing users"):
            # 合并用户所有交互记录为一条
            unique_items = group.drop_duplicates(subset=['sid'])
            user_history[uid] = [
                {'sid': row['sid'], 'timestamp': row['timestamp']} 
                for _, row in unique_items.sort_values('timestamp').iterrows()
            ]
        
        return user_history

    def generate_basic_prompts(self, dataset, sample_rate=0.1, num_prompts=5):
        """
        生成基础prompt，每个用户一条记录
        候选商品基于全局访问频率采样
        """
        user_history = dataset['user_history']
        meta = dataset['meta']
        
        # 计算商品全局访问频率
        item_freq = defaultdict(int)
        for uid, items in user_history.items():
            for item in items:
                clean_id = item['sid'].replace('i', '')
                item_freq[clean_id] += 1
        
        all_items = list(item_freq.keys())
        weights = np.array([item_freq[item] for item in all_items], dtype=np.float32)
        weights /= weights.sum()
        
        num_candidates = max(1, int(len(all_items) * sample_rate))
        
        prompts = []
        for uid, items in user_history.items():
            if not items:
                continue
                
            # 构建用户历史记录（去重后）
            history = []
            seen_items = set()
            for item in sorted(items, key=lambda x: x['timestamp']):
                clean_id = item['sid'].replace('i', '')
                if clean_id not in seen_items:
                    seen_items.add(clean_id)
                    history.append({
                        'item_id': clean_id,
                        'title': meta.get(clean_id, "未知商品"),
                        'timestamp': item['timestamp']
                    })
            
            # 基于频率采样候选商品
            candidate_indices = np.random.choice(
                len(all_items), 
                size=num_candidates, 
                replace=False, 
                p=weights
            )
            candidates = []
            for idx in candidate_indices:
                item_id = all_items[idx]
                candidates.append({
                    'item_id': item_id,
                    'title': meta.get(item_id, "未知商品"),
                    'frequency': item_freq[item_id]
                })
            
            prompts.append({
                'user_id': uid,
                'history': history,
                'candidates': candidates
            })
        
        # 随机选择指定数量的prompt
        if len(prompts) > num_prompts:
            prompts = random.sample(prompts, num_prompts)
        
        return prompts

    def simulate_timed_requests(self, dataset, num_requests=10, time_scale=1.0):
        """
        模拟带时间戳的请求流
        1. 第一条立即发送
        2. 后续请求基于前一条的相对时间间隔
        """
        prompts = self.generate_basic_prompts(dataset, sample_rate=0.1, num_prompts=num_requests)
        
        # 分析全局时间间隔分布
        all_intervals = []
        for uid, items in dataset['user_history'].items():
            timestamps = [item['timestamp'] for item in items]
            if len(timestamps) > 1:
                intervals = np.diff(sorted(timestamps))
                all_intervals.extend(intervals)
        
        if all_intervals:
            global_mean = np.mean(all_intervals)
            global_std = np.std(all_intervals)
        else:
            global_mean = 86400
            global_std = 43200
        
        # 为每个prompt计算时间间隔
        intervals = []
        for prompt in prompts:
            if prompt['history']:
                timestamps = [item['timestamp'] for item in prompt['history']]
                if len(timestamps) > 1:
                    user_intervals = np.diff(sorted(timestamps))
                    mean_int = np.mean(user_intervals)
                    std_int = np.std(user_intervals)
                else:
                    mean_int = global_mean
                    std_int = global_std
            else:
                mean_int = global_mean
                std_int = global_std
            
            interval = max(0, np.random.normal(
                mean_int * time_scale,
                std_int * time_scale
            ))
            intervals.append(interval)
        
        # 第一条立即发送(间隔设为0)
        intervals[0] = 0
        
        # 模拟请求发射
        start_time = time.time()
        last_time = start_time
        
        for i, (prompt, interval) in enumerate(zip(prompts, intervals), 1):
            # 计算等待时间（相对于上一条的间隔）
            wait_time = max(0, interval - (time.time() - last_time))
            time.sleep(wait_time)
            
            # 记录实际发射时间
            actual_time = time.time()
            prompt['request_time'] = actual_time
            prompt['request_time_str'] = datetime.fromtimestamp(actual_time).strftime('%Y-%m-%d %H:%M:%S')
            prompt['time_interval'] = interval
            
            # 计算实际间隔
            actual_interval = actual_time - last_time
            last_time = actual_time
            
            # 打印发射信息
            print(f"\n请求 {i}/{len(prompts)} 在 {prompt['request_time_str']} 发射")
            print(f"预设间隔: {interval:.1f}s (≈{interval/3600:.2f}小时)")
            print(f"实际间隔: {actual_interval:.1f}s (≈{actual_interval/3600:.2f}小时)")
            print("-" * 50)
            print(f"用户ID: {prompt['user_id']}")
            print("\n用户历史记录:")
            print(self._format_history(prompt['history']))
            #print("\n候选商品列表:")
            #print(self._format_candidates(prompt['candidates']))
            print("=" * 60)
        
        return prompts

    def _format_history(self, history):
        """格式化用户历史记录为字符串"""
        lines = []
        for i, item in enumerate(history, 1):
            dt = datetime.fromtimestamp(item['timestamp']) if item['timestamp'] > 0 else "无时间戳"
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt, datetime) else str(dt)
            lines.append(
                f"{i}. 商品ID: {item['item_id']}\n"
                f"   商品标题: {item['title']}\n"
                f"   访问时间: {time_str}"
            )
        return "\n".join(lines)
    
    def _format_candidates(self, candidates):
        """格式化候选商品为字符串"""
        lines = []
        for i, item in enumerate(candidates, 1):
            lines.append(
                f"{i}. 候选商品ID: {item['item_id']}\n"
                f"   商品标题: {item['title']}\n"
                f"   历史访问频率: {item['frequency']}次"
            )
        return "\n".join(lines)

    def print_basic_prompts(self, prompts, num_to_print=1):
        """打印基础prompt示例"""
        print("\n基础Prompt示例:")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts[:num_to_print], 1):
            print(f"\nPrompt {i}/{len(prompts)}")
            print("-" * 50)
            print(f"用户ID: {prompt['user_id']}")
            print("\n用户历史记录:")
            print(self._format_history(prompt['history']))
            print("\n候选商品列表:")
            print(self._format_candidates(prompt['candidates']))
            print("=" * 60)


if __name__ == "__main__":
    dataset_processor = TianmaoDataset(min_uc=1, min_sc=1)
    
    try:
        dataset = dataset_processor.process()
        
        print("\nProcessed dataset structure:")
        print(f"- Total users: {dataset['statistics']['num_users']}")
        print(f"- Total items: {dataset['statistics']['num_items']}")
        print(f"- Total interactions: {dataset['statistics']['total_interactions']}")
        
        print("\n测试基础prompt生成...")
        basic_prompts = dataset_processor.generate_basic_prompts(dataset, sample_rate=0.1, num_prompts=3)
        dataset_processor.print_basic_prompts(basic_prompts, num_to_print=2)
        
        print("\n测试时间戳模拟请求...")
        dataset_processor.simulate_timed_requests(dataset, num_requests=5, time_scale=1e-5)
            
    except Exception as e:
        print(f"\nError processing dataset: {str(e)}")
        raise