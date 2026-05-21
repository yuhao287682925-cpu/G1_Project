import pickle
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="提取并计算 AMP 数据 (兼容 G1_Project)")
    parser.add_argument("--input", type=str, default="XingJiang002.pkl", help="输入的 pkl 动作文件路径")
    parser.add_argument("--output", type=str, default="xinjiang_amp_fixed.npz", help="输出的 npz 动作文件路径")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    # 加载原始 PKL
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # 提取数组 (确保维度为 [Frames, 29])
    # X 为前方，Z 为上方
    dof_pos = data['dof_pos'] 
    root_pos = data['root_pos']
    root_rot = data['root_rot']

    # 获取或计算时间步长 dt
    fps = data.get('fps', 50.0)
    dt = 1.0 / fps

    # 使用中心差分计算速度 (dof_vel 和 root_lin_vel)
    # np.gradient 沿 axis=0 (帧) 对所有坐标轴进行求导
    dof_vel = np.gradient(dof_pos, axis=0) / dt
    root_lin_vel = np.gradient(root_pos, axis=0) / dt

    # 使用 savez 保存为非 object 格式
    np.savez(output_path, 
             dof_pos=dof_pos.astype(np.float32), 
             dof_vel=dof_vel.astype(np.float32),
             root_pos=root_pos.astype(np.float32),
             root_rot=root_rot.astype(np.float32),
             root_lin_vel=root_lin_vel.astype(np.float32))
             
    print(f"数据处理完毕，共处理 {dof_pos.shape[0]} 帧数据。")
    print(f"帧率(fps): {fps}, 计算时间差(dt): {dt:.4f}s")
    print(f"已保存为兼容性最强的 .npz 格式至: {output_path}")

if __name__ == "__main__":
    main()