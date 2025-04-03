import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

class DataProcessor:
    """Data processing class for loading, processing and feature extraction"""
    
    def __init__(self, data_dir: str):
        """
        Initialize data processor
        
        Parameters:
            data_dir: Data directory path
        """
        self.data_dir = data_dir
        self.files = self._get_csv_files()
        
    def _get_csv_files(self) -> List[str]:
        """Get all CSV file paths"""
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv') and not file.startswith('.'):
                files.append(os.path.join(self.data_dir, file))
        return files
    
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load and process all CSV file data
        
        Returns:
            Processed dataframe
        """
        all_data = []
        
        for file_path in self.files:
            try:
                df = pd.read_csv(file_path)
                # 添加文件标识
                file_name = os.path.basename(file_path)
                df['file_name'] = file_name
                all_data.append(df)
            except Exception as e:
                print(f"Failed to read file: {file_path}, error: {e}")
                
        if not all_data:
            raise ValueError("No valid data files found")
            
        return pd.concat(all_data, ignore_index=True)
    
    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sequence features from a single file
        
        Parameters:
            df: Dataframe of a single file
            
        Returns:
            Dataframe containing extracted features
        """
        # 按文件名和browsing_mode分组
        grouped = df.groupby(['file_name', 'browsing_mode'])
        
        sequence_features = []
        
        for (file_name, browsing_mode), group in grouped:
            # 对每个序列提取特征
            
            # 1. 计算动作类型分布
            type_counts = group['type'].value_counts().to_dict()
            type_0_count = type_counts.get(0, 0)
            type_1_count = type_counts.get(1, 0)
            type_2_count = type_counts.get(2, 0)
            
            # 2. 计算动作间隔时间特征
            if len(group) > 1:
                # 按时间排序
                sorted_group = group.sort_values('start_time')
                # 计算时间间隔
                time_diffs = sorted_group['start_time'].diff().dropna().values
                mean_time_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 0
                std_time_interval = np.std(time_diffs) if len(time_diffs) > 0 else 0
                max_time_interval = np.max(time_diffs) if len(time_diffs) > 0 else 0
                min_time_interval = np.min(time_diffs) if len(time_diffs) > 0 else 0
            else:
                mean_time_interval = std_time_interval = max_time_interval = min_time_interval = 0
            
            # 3. 计算手势区域分布
            gesture_area_counts = group['gesture_area'].value_counts(normalize=True).to_dict()
            
            # 4. 计算动作特征的统计量
            mean_velocity_x = group['velocity_x'].mean()
            mean_velocity_y = group['velocity_y'].mean()
            std_velocity_x = group['velocity_x'].std()
            std_velocity_y = group['velocity_y'].std()
            mean_distance = group['total_distance'].mean()
            std_distance = group['total_distance'].std()
            mean_duration = group['duration'].mean()
            std_duration = group['duration'].std()
            
            # 5. 计算动作类型转换模式
            if len(group) > 1:
                sorted_group = group.sort_values('start_time')
                type_transitions = sorted_group['type'].astype(str).str.cat(sep='')
                # 计算常见模式出现的次数
                swipe_tap_count = type_transitions.count('10')  # 滑动后点击
                tap_swipe_count = type_transitions.count('01')  # 点击后滑动
                long_press_count = type_transitions.count('2')  # 长按
            else:
                swipe_tap_count = tap_swipe_count = long_press_count = 0
            
            # 6. 计算动作区域特征
            gesture_area_mode = group['gesture_area'].mode().iloc[0] if not group['gesture_area'].empty else 0
            
            # 7. 计算运动传感器数据统计量
            mean_motion_x = group['motion_x'].mean()
            mean_motion_y = group['motion_y'].mean()
            mean_motion_z = group['motion_z'].mean()
            std_motion_x = group['motion_x'].std()
            std_motion_y = group['motion_y'].std()
            std_motion_z = group['motion_z'].std()
            
            # 8. 计算序列长度
            sequence_length = len(group)
            
            # 创建特征字典
            features = {
                'file_name': file_name,
                'browsing_mode': browsing_mode,
                'type_0_count': type_0_count,
                'type_1_count': type_1_count,
                'type_2_count': type_2_count,
                'mean_time_interval': mean_time_interval,
                'std_time_interval': std_time_interval,
                'max_time_interval': max_time_interval,
                'min_time_interval': min_time_interval,
                'mean_velocity_x': mean_velocity_x,
                'mean_velocity_y': mean_velocity_y,
                'std_velocity_x': std_velocity_x,
                'std_velocity_y': std_velocity_y,
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'mean_duration': mean_duration,
                'std_duration': std_duration,
                'swipe_tap_count': swipe_tap_count,
                'tap_swipe_count': tap_swipe_count,
                'long_press_count': long_press_count,
                'gesture_area_mode': gesture_area_mode,
                'mean_motion_x': mean_motion_x,
                'mean_motion_y': mean_motion_y,
                'mean_motion_z': mean_motion_z,
                'std_motion_x': std_motion_x,
                'std_motion_y': std_motion_y,
                'std_motion_z': std_motion_z,
                'sequence_length': sequence_length
            }
            
            # 添加手势区域分布特征
            for area in range(9):  # 假设手势区域从0到8
                features[f'gesture_area_{area}'] = gesture_area_counts.get(area, 0)
            
            sequence_features.append(features)
        
        return pd.DataFrame(sequence_features)
    
    def extract_time_series_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract time series features from a single file
        
        Parameters:
            df: Dataframe of a single file
            
        Returns:
            Dictionary containing sequences and labels
        """
        # 按文件名分组
        grouped = df.groupby('file_name')
        
        sequences = []
        labels = []
        file_names = []
        
        # 用于序列特征的列
        feature_cols = [
            'type', 'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
            'total_distance', 'duration', 'motion_x', 'motion_y', 'motion_z', 'gesture_area'
        ]
        
        # 定义最大序列长度
        max_seq_length = 50
        
        for file_name, group in grouped:
            # 获取browsing_mode（假设一个文件只有一种模式）
            browsing_mode = group['browsing_mode'].iloc[0]
            
            # 重新分类：只保留类别1和2，其他归为类别3
            if browsing_mode not in [1, 2]:
                browsing_mode = 3
            
            # 按时间排序
            sorted_group = group.sort_values('start_time')
            
            # 提取特征序列
            seq_features = sorted_group[feature_cols].values
            
            # 归一化数值特征（除了type和gesture_area）
            numerical_cols = ['velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
                           'total_distance', 'duration', 'motion_x', 'motion_y', 'motion_z']
            numerical_indices = [feature_cols.index(col) for col in numerical_cols]
            
            # 对每个数值特征列进行归一化
            for idx in numerical_indices:
                # 如果特征列有非零值，进行归一化
                if np.max(np.abs(seq_features[:, idx])) > 0:
                    seq_features[:, idx] = seq_features[:, idx] / np.max(np.abs(seq_features[:, idx]))
            
            # 截断或填充序列到固定长度
            if len(seq_features) > max_seq_length:
                seq_features = seq_features[:max_seq_length]
            elif len(seq_features) < max_seq_length:
                # 填充零到最大长度
                padding = np.zeros((max_seq_length - len(seq_features), len(feature_cols)))
                seq_features = np.vstack([seq_features, padding])
            
            sequences.append(seq_features)
            labels.append(browsing_mode)
            file_names.append(file_name)
        
        return {
            'sequences': np.array(sequences),
            'labels': np.array(labels),
            'file_names': np.array(file_names)
        }
    
    def prepare_data(self, use_time_series: bool = False) -> Tuple:
        """
        Prepare data for model training
        
        Parameters:
            use_time_series: Whether to use time series features
            
        Returns:
            If use_time_series is True, returns sequence features and labels;
            otherwise returns extracted feature dataframe
        """
        # 加载所有数据
        all_data = self.load_and_process_data()
        
        if use_time_series:
            # 提取时间序列特征
            return self.extract_time_series_features(all_data)
        else:
            # 提取序列特征
            return self.extract_sequence_features(all_data) 