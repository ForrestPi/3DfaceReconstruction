## 1. preparing

**环境安装：**  安装Eigen库
**DeformationTransfer编译：** `cd modules/deformation/cython`  `python3 setup.py build_ext -i `.

## 2. pipeline
`cd tools`
files in the tools:

- `extract_frame.py`: 抽取视频帧
- `to_3dmm.py`: 人脸转换成3DMM，交换表情主成分系数实现直接换脸
- `swap.py`: deformationTranfer换脸，有4个数据输入，sourceA，targetA，sources，targets
- `combine_img.py`: 组合图像，便于对比观察
- `generate_video.sh`: 用ffmpeg将图像转成视频  

## 3. references 
1. https://github.com/YadiraF/face3d
2. https://github.com/cleardusk/3DDFA
3. https://github.com/chand81/Deformation-Transfer
