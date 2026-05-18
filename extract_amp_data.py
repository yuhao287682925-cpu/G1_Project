import pickle
import numpy as np

# 加载原始 PKL
with open('/home/robot/G1_Project/XingJiang002.pkl', 'rb') as f:
    data = pickle.load(f)

# 提取数组 (确保维度为 [Frames, 29])
# 再次提醒：X 为前方，Z 为上方
dof_pos = data['dof_pos'] 
dof_vel = np.zeros_like(dof_pos) # 建议按之前逻辑计算速度

# 使用 savez 保存为非 object 格式
np.savez('/home/robot/G1_Project/xinjiang_amp_fixed.npz', 
         dof_pos=dof_pos.astype(np.float32), 
         dof_vel=dof_vel.astype(np.float32),
         root_pos=data['root_pos'].astype(np.float32),
         root_rot=data['root_rot'].astype(np.float32))
print("已保存为兼容性最强的 .npz 格式")