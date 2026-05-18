from setuptools import setup, find_packages

setup(
    name="unitree_rl_lab",
    version="1.0",
    # 核心：告诉它包在 source/unitree_rl_lab 文件夹里
    package_dir={"": "source/unitree_rl_lab"},
    packages=find_packages(where="source/unitree_rl_lab"),
    install_requires=[
        'numpy',
        'torch',
        'mujoco',
    ],
)
