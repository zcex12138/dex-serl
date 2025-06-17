import numpy as np

# 加载原始数据
data = np.load('./data/test_recording.npz', allow_pickle=True)

data_raw = data['data']
data_raw['total_time'] = 10.0
# 重命名键（假设旧键是 'keypoints'，新键是 'new_key'）
data['total_time'] = 10.0
# 保存修改后的数据（使用压缩格式节省空间）
np.savez_compressed('./data/test_recording_renamed.npz', **data, total_time=10.0)