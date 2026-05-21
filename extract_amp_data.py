import pickle
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract AMP data from pkl to csv for csv_to_npz.py")
    parser.add_argument("--input", type=str, required=True, help="Input pkl file path")
    parser.add_argument("--output", type=str, required=True, help="Output csv file path")
    args = parser.parse_args()

    # 读取pkl文件
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # 提取所需的基础动作数据
    dof_pos = data['dof_pos']      # (T, 29)
    root_pos = data['root_pos']    # (T, 3)
    root_rot = data['root_rot']    # (T, 4) in xyzw format
    fps = data.get('fps', 50.0)
    
    print(f"[INFO] Loaded {args.input}: {root_pos.shape[0]} frames at {fps} FPS")

    # CSV格式要求: Nx36
    # [root_pos_x, root_pos_y, root_pos_z, root_rot_x, root_rot_y, root_rot_z, root_rot_w, dof_0...dof_28]
    csv_data = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    np.savetxt(args.output, csv_data, delimiter=",", fmt="%.6f")
    print(f"[SUCCESS] CSV saved to {args.output}")
    print(f"\n--- NEXT STEP ON LINUX ---")
    print(f"To get the final `.npz` file with full body kinematics tracking (body_pos_w), run:")
    print(f"python unitree_rl_lab/scripts/mimic/csv_to_npz.py -f {args.output} --input_fps {int(fps)}")

if __name__ == "__main__":
    main()