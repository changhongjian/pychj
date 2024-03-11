
# 安装记录
```bash
python setup.py clean --all
python setup.py develop
pip install --upgrade git+https://username:token@xx.com/username/pychj.git
pip install git+https://github.com/changhongjian/pychj.git
#pip install --upgrade --force-reinstall <package>
#pip install -I <package>
#pip install --ignore-installed <package>
```

# OLD
`tree | egrep -v '__pycache__|.pyc|.dll|.pyd'`

```
.
|-- README.md
|-- base
|   |-- clip.py
|   |-- file.py
|   |-- obj.py
|   `-- sys.py
|-- comm
|   |-- data_handle.py
|   |-- face_model_compute.py
|   |-- geometry.py
|   |-- geometry2d.py
|   |-- geometry_ext.py
|   |-- include_full.py
|   |-- pic.py
|   |-- plt.py
|   |-- render.py
|   `-- video.py
|-- libext
|   |-- console.py
|   |-- exe
|   |   |-- ffmpeg.py
|   |   `-- mediainfo.py
|   `-- ogl
|       |-- light.py
|       `-- objloader.py
|-- math
|   |-- geometry
|   |   |-- BFM.py
|   |   |-- lie_algebra.py
|   |   |-- light.py
|   |   |-- points.py
|   |   `-- rot.py
|   |-- img
|   |   `-- base.py
|   |-- solver.py
|   `-- type_convert.py
|-- nnet
|   |-- nnet_common
|   |   |-- configure_utils.py
|   |   |-- data_handle.py
|   |   |-- data_handle_v2.py
|   |   |-- include.py
|   |   |-- mobilenet_chj.py
|   |   |-- mobilenet_v1.py
|   |   |-- net_utils.py
|   |   `-- nnet_comm.py
|   `-- utils
|       |-- data_hd.py
|       |-- hparams.py
|       |-- loss.py
|       `-- render.py
|-- programs
|   |-- chj-avi_to_mp4.sh
|   |-- chj-mesh_get_obj_vids_by_f
|   `-- chj-obj_add_white
|-- projects
|   |-- bfm_fit
|   |   |-- bfm.py
|   |   |-- bfm_solve.py
|   |   |-- help_key_fit.py
|   |   |-- key_frame_use.py
|   |   |-- ldmk_table.py
|   |   `-- transform.py
|       |-- cpu_render.py
|       `-- gpu_render.py
|-- speed
|   |-- help.py
|   `-- split_run.py
    |-- CMakeLists.txt
    |-- src
    |   |-- Interface.cpp
    |   |-- Interface.h
    |   |-- common_math.h
    |   `-- shape_visual.cpp
    |-- src_cu
    |   |-- blas_kernels.cu
    |   |-- common_math_gpu.h
    |   |-- cuda_tools.c
    |   |-- cuda_tools.h
    |   |-- make_render.cu
    |   |-- render_infos.cu
    |   |-- render_math.h
    |   `-- render_uv.cu

```

# 需要安装的库

```bash
conda env list
conda create -n pytorch1.3 python=3.7
conda activate pytorch1.3
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Pillow
pip install scipy
pip install matplotlib
pip install tqdm
pip install pyyaml
pip install easydict
```


