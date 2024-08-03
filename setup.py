from setuptools import setup, find_packages

setup(
    name='chj',
    packages=find_packages(),
    version='0.1.8',
    description="Pylib CHJ Package",
    install_requires=[          # 添加了依赖的 package
        'opencv-python-headless',
        'Pillow',
        'scipy',
        'matplotlib',
        'tqdm',
        'pyyaml',
        'ruamel.yaml',
        'easydict',
        'torch',
        'imageio'
    ]
)
