
---
# 基于 Wav2Lip-GFPGAN 深度学习模型的数字人Demo


***
+ 工作中遇到简单整理
+ 理解不足小伙伴帮忙指正


**<font color="009688"> 对每个人而言，真正的职责只有一个：找到自我。然后在心中坚守其一生，全心全意，永不停息。所有其它的路都是不完整的，是人的逃避方式，是对大众理想的懦弱回归，是随波逐流，是对内心的恐惧 ——赫尔曼·黑塞《德米安》**</font>

***

## 简单介绍
### Wav2Lip-GAN

`Wav2Lip-GAN` 是一种基于生成对抗网络（GAN）的语音到唇形的转换模型。[https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

基本原理是使用语音信号和人脸图像来训练一个生成器网络，该网络可以将输入的语音信号转换为对应的唇形。

该模型包括两个子网络：
+ 一个是语音识别网络，用于将语音信号转换为文本；
+ 另一个是唇形生成网络，用于将文本和人脸图像作为输入，生成对应的唇形。

两个网络通过GAN框架进行训练，以使生成的唇形尽可能地逼真。在测试阶段，给定一个语音信号和一个人脸图像，该模型可以生成一个与语音信号相对应的唇形序列，从而实现语音到唇形的转换。

### GFPGAN

腾讯 `GFPGAN` 是一种基于生成对抗网络（GAN）的图像超分辨率模型。[https://github.com/TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)

基本原理是使用低分辨率的图像作为输入，通过生成器网络将其转换为高分辨率的图像。

该模型包括两个子网络：
+ 一个是生成器网络，用于将低分辨率图像转换为高分辨率图像；
+ 另一个是判别器网络，用于评估生成的图像是否逼真。

两个网络通过GAN框架进行训练，以使生成的图像尽可能地接近真实图像。在测试阶段，给定一个低分辨率的图像，该模型可以生成一个与之对应的高分辨率图像。腾讯GFPGAN采用了一些创新的技术，如渐进式训练、自适应实例归一化等，使得其在图像超分辨率任务中表现出色。

Demo 来自项目 [https://github.com/ajay-sainy/Wav2Lip-GFPGAN/](https://github.com/ajay-sainy/Wav2Lip-GFPGAN/) 完成，小伙伴可以直接参考。作者提供了一个`ipynb` Demo `GitHub\Wav2Lip-GFPGAN\Wav2Lip-GFPGAN.ipynb`,有基础的小伙伴按照步骤即可完成，下面的不需要看

![在这里插入图片描述](https://img-blog.csdnimg.cn/e45b23dfac854dbf8a7af03db5fbc858.png)

这里完整的搭建步骤，和涉及的素材/脚本以上传git库，小伙伴可以直接克隆按照文档操作运行

[https://github.com/LIRUILONGS/Wav2Lip-GFPGAN_Python_Demo](https://github.com/LIRUILONGS/Wav2Lip-GFPGAN_Python_Demo)


# 原 readme.md

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajay-sainy/Wav2Lip-GFPGAN/blob/main/Wav2Lip-GFPGAN.ipynb)



Combine Lip Sync AI and Face Restoration AI to get ultra high quality videos.

[Demo Video](https://www.youtube.com/watch?v=jArkTgAMA4g)  

[![Demo Video](https://img.youtube.com/vi/jArkTgAMA4g/default.jpg)](https://youtu.be/jArkTgAMA4g)

Projects referred:
1. https://github.com/Rudrabha/Wav2Lip
2. https://github.com/TencentARC/GFPGAN

Video sources:  
1. https://www.youtube.com/watch?v=39w_zYB7AVM&t=0s
2. https://www.youtube.com/watch?v=LQCQym6hVMo&t=0s


# 环境搭建记录

需要的模型文件提前下载

## 涉及到的模型和安装包下载

###　Wav2Lip

可以在项目中看到下载路径: [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

`Wav2Lip`：[https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)


`Wav2Lip + GAN`　：[https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)



`ffmpeg`: [https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z) ,Linux 环境直接用包管理工具安装即可

ffmpeg 装完之后 win系统 需要配置环境变量，这里不多讲。


### GFPGAN

`GFPGANv1.3.pth`:[https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

`parsing_parsenet.pth`:[https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth)

`detection_Resnet50_Final.pth`:[https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth)

## 环境安装

### wav2lip 环境
当前系统环境为 `window11,Anaconda3` 使用CPU 跑，虚拟环境创建
```bash
C:\Users\liruilong>conda create -n wav2lip python=3.8
C:\Users\liruilong>conda info --envs
# conda environments:
#
base                  *  C:\ProgramData\Anaconda3
myenv                    C:\Users\liruilong\AppData\Local\conda\conda\envs\myenv
wav2lip                  C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip
```
切换虚拟环境的时候，报错了
```bash
C:\Users\liruilong>conda activate wav2lip
.....
```
后来在`Anaconda Prompt (Anaconda3)` 可以正常执行

```bash
(base) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>conda activate wav2lip

(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>conda list
.....
```
安装 `requirements.txt` 中的依赖库，直接安装报错了
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>pip install -r requirements.txt   -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
Looking in indexes: http://pypi.douban.com/simple/
```
需要添加 `--use-pep517`
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>pip install -r requirements.txt   -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com  --use-pep517
Looking in indexes: http://pypi.douban.com/simple/
```
检测 `wav2lip` 环境运行Demo 测试一下，当前项目预留了一些素材，这里使用模型`wav2lip.pth`
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>python .\Wav2Lip-master\inference.py --checkpoint_path .\Wav2Lip-master\checkpoints\wav2lip.pth --face .\inputs\kim_7s_raw.mp4 --audio .\inputs\kim_audio.mp3 --outfile result.mp4
Using cpu for inference.
Reading video frames...
Number of frames available for inference: 223
Extracting raw audio...
...................................
[libx264 @ 0000022caf538200] Weighted P-Frames: Y:1.2% UV:1.2%
[libx264 @ 0000022caf538200] ref P L0: 68.7%  8.6% 16.2%  6.4%
[libx264 @ 0000022caf538200] ref B L0: 75.0% 20.2%  4.8%
[libx264 @ 0000022caf538200] ref B L1: 94.9%  5.1%
[libx264 @ 0000022caf538200] kb/s:1433.66
[aac @ 0000022caf528940] Qavg: 237.868
```
运行完会在当前目录生成 `result.mp4` 文件

[https://www.bilibili.com/video/BV1fX4y187jW/](https://www.bilibili.com/video/BV1fX4y187jW/)

然后用模型`wav2lip_gan.pth` 在试下
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>python .\Wav2Lip-master\inference.py --checkpoint_path  .\inputs\wav2lip_gan.pth --face .\inputs\kim_7s_raw.mp4 --audio .\inputs\kim_audio.mp3 --outfile result.mp4
Using cpu for inference.
```
[https://www.bilibili.com/video/BV1Vo4y1T7F2/](https://www.bilibili.com/video/BV1Vo4y1T7F2/)

这里 wav2lip 环境已经安装完成

### GFPGAN 环境

准备一个新的音视频，使用 `wav2lip_gan` 生成，准备GFPGAN 环境
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>python .\Wav2Lip-master\inference.py --checkpoint_path  .\inputs\wav2lip_gan.pth --face .\inputs\demo.mp4 --audio .\inputs\demo_5_y.mp3 --outfile result.mp4
Using cpu for inference.
Reading video frames...
Number of frames available for inference: 2116
Extracting raw audio..
。。。。。。。。。。。。。。。。。。。。。
[libx264 @ 000001ba2a798d80] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 18% 18% 48%  3%  2%  2%  2%  3%  3%
[libx264 @ 000001ba2a798d80] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 23% 22% 17%  6%  6%  6%  6%  7%  8%
[libx264 @ 000001ba2a798d80] i8c dc,h,v,p: 49% 20% 22%  8%
[libx264 @ 000001ba2a798d80] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 000001ba2a798d80] ref P L0: 80.9% 10.0%  6.6%  2.5%
[libx264 @ 000001ba2a798d80] ref B L0: 87.8% 10.5%  1.7%
[libx264 @ 000001ba2a798d80] ref B L1: 98.7%  1.3%
[libx264 @ 000001ba2a798d80] kb/s:703.37
[aac @ 000001ba2a79a780] Qavg: 170.234

(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>
```
[https://www.bilibili.com/video/BV1cX4y1h7k8/](https://www.bilibili.com/video/BV1cX4y1h7k8/)


创建一个结果文件夹
```bash
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN> mkdir results


    目录: C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----          2023/6/9      7:14                results


PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>
```
需要把上面生成的文件移到这个文件夹里面，然后执行下面的脚本
```py
# day1.py

wav2lipFolderName = 'Wav2Lip-master'
gfpganFolderName = 'GFPGAN-master'
wav2lipPath =  '.\\' + wav2lipFolderName
gfpganPath = '.\\' + gfpganFolderName
outputPath = ".\\results"

import cv2
from tqdm import tqdm
from os import path

import os

# 上一步生成的视频
inputVideoPath = outputPath+'\\result.mp4'
# 中间数据
unProcessedFramesFolderPath = outputPath+'\\frames'

if not os.path.exists(unProcessedFramesFolderPath):
  os.makedirs(unProcessedFramesFolderPath)

vidcap = cv2.VideoCapture(inputVideoPath)
numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):
    _,image = vidcap.read()
    cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4)+'.jpg'), image)

print("unProcessedFramesFolderPath:",unProcessedFramesFolderPath)
print("inputVideoPath:",inputVideoPath)

```
作用是将wav2lip处理的视频按帧数逐帧读取，将每一帧保存为 JPEG 格式的图片，并将这些图片保存到指定的文件夹 `unProcessedFramesFolderPath` 中
```bash
(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN>python day1.py
FPS:  25.0 Frames:  1793
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1793/1793 [00:10<00:00, 166.99it/s]
unProcessedFramesFolderPath:  
inputVideoPath: .\results\result.mp4

(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN>
```
之后会在 `.\results\frames` 看到切好的照片


现在准备 GFPGAN-master 的环境

```bash
(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com --use-pep517
Looking in indexes: http://pypi.douban.com/simple/
..........
Installing collected packages: numpy, scikit-image
  Attempting uninstall: numpy
    Found existing installation: numpy 1.23.5
    Uninstalling numpy-1.23.5:
      Successfully uninstalled numpy-1.23.5
  Attempting uninstall: scikit-image
    Found existing installation: scikit-image 0.20.0
    Uninstalling scikit-image-0.20.0:
      Successfully uninstalled scikit-image-0.20.0
Successfully installed numpy-1.20.3 scikit-image-0.19.3

(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>
```

GFPGANv1.3.pth 模型放到 `/experiments/pretrained_models` 目录下
```bash
(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>mkdir -p .\\experiments\pretrained_models

(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>cd  .\\experiments\pretrained_models
```
确认模型
```bash
    目录: C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master\experiments\pretrained_models


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----          2023/6/7      1:43      348632874 GFPGANv1.3.pth
```

之后执行下面的命令,处理图片
```bash
python inference_gfpgan.py -i $unProcessedFramesFolderPath -o $outputPath -v 1.3 -s 2 --only_center_face --bg_upsampler None
```
替换对应的变量，如果模型无法下载，需要把前面下载的放到指定位置
```bash
(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>python inference_gfpgan.py -i ..\results\frames -o ..\results -v 1.3 -s 2 --only_center_face --bg_upsampler None
C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip\lib\site-packages\torchvision\transforms\functional_tensor.py:5: UserWarning: The torchvision.transforms.functional_tensor module is deprecated in 0.15 and will be **removed in 0.17**. Please don't rely on it. You probably just need to use APIs in torchvision.transforms.functional or in torchvision.transforms.v2.functional.
  warnings.warn(
C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" to C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip\lib\site-packages\facexlib\weights\detection_Resnet50_Final.pth

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104M/104M [00:06<00:00, 16.1MB/s]
Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" to C:\Users\liruilong\AppData\Local\conda\conda\envs\wav2lip\lib\site-packages\facexlib\weights\parsing_parsenet.pth

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81.4M/81.4M [00:05<00:00, 14.8MB/s]
0it [00:00, ?it/s]
  warnings.warn(msg)
  0%|                                                                                                                                                                                  | 0/1793 [00:00<?, ?it/s]Processing 0000.jpg ...
  0%|                                                                                                                                                                        | 1/1793 [00:06<3:18:38,  6.65s/it]Processing 0001.jpg ...
  0%|▏                                                                                                                                                                       | 2/1793 [00:13<3:18:06,  6.64s/it]P
...............................
(wav2lip) C:\Users\liruilong\Documents\GitHub\Wav2Lip-GFPGAN\GFPGAN-master>
```
OK 跑完之后，需要用处理的图片合成视频，执行下面的脚本
```py


import os


outputPath = ".\\results"

restoredFramesPath = outputPath + '\\restored_imgs\\'
processedVideoOutputPath = outputPath

dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

import cv2
import numpy as np

batch = 0
batchSize = 300
from tqdm import tqdm
for i in tqdm(range(0, len(dir_list), batchSize)):
  img_array = []
  start, end = i, i+batchSize
  print("processing ", start, end)
  for filename in  tqdm(dir_list[start:end]):
      filename = restoredFramesPath+filename;
      img = cv2.imread(filename)
      if img is None:
        continue
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)


  out = cv2.VideoWriter(processedVideoOutputPath+'\\batch_'+str(batch).zfill(4)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
  batch = batch + 1
 
  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()

concatTextFilePath = outputPath + "\\concat.txt"
concatTextFile=open(concatTextFilePath,"w")
for ips in range(batch):
  concatTextFile.write("file batch_" + str(ips).zfill(4) + ".avi\n")
concatTextFile.close()

concatedVideoOutputPath = outputPath + "\\concated_output.avi"
print("concatedVideoOutputPath:",concatedVideoOutputPath)

finalProcessedOuputVideo = processedVideoOutputPath+'\\final_with_audio.avi'
print("finalProcessedOuputVideo:",finalProcessedOuputVideo)
# ffmpeg -y -f concat -i {concatTextFilePath} -c copy {concatedVideoOutputPath} 

#ffmpeg -y -i {concatedVideoOutputPath} -i {inputAudioPath} -map 0 -map 1:a -c:v copy -shortest {finalProcessedOuputVideo}

#from google.colab import files
#files.download(finalProcessedOuputVideo)

```
```bash
(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>python day2.py
  0%|                                                                                                                                                                                     | 0/6 [00:00<?, ?it/s]processing  0 300

  0%|                                                                                                                                                                                   | 0/300 [00:00<?, ?it/s]
  4%|██████▏                                                                                                                                                                  | 11/300 [00:00<00:02, 107.59it/s]
  7%|███████████▊                                                                                                                                                             | 21/300 [00:00<00:02, 104.49it/s]
 11%|██████████████████
 ...................
 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 293/293 [00:02<00:00, 107.10it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:25<00:00,  4.26s/it]
concatedVideoOutputPath: .\results\concated_output.avi
finalProcessedOuputVideo: .\results\final_with_audio.avi

(wav2lip) C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN>
```
使用 ffmpeg 合并视频

```bash
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN> cd .\results\
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\results> ffmpeg -y -f concat -i .\concat.txt  -c copy .\concated_output.avi
.....................
frame= 1793 fps=0.0 q=-1.0 Lsize=   24625kB time=00:00:59.76 bitrate=3375.3kbits/s speed=1.76e+03x
video:24577kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.197566%
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\results> ls


    目录: C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\results


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----          2023/6/9      7:25                frames
d-----          2023/6/9     11:03                restored_imgs
-a----          2023/6/9     11:42        4231050 batch_0000.avi
-a----          2023/6/9     11:42        4274254 batch_0001.avi
-a----          2023/6/9     11:42        4281898 batch_0002.avi
-a----          2023/6/9     11:42        4165970 batch_0003.avi
-a----          2023/6/9     11:42        4222324 batch_0004.avi
-a----          2023/6/9     11:42        4069836 batch_0005.avi
-a----          2023/6/9     11:42            126 concat.txt
-a----          2023/6/9     11:52       25216450 concated_output.avi
-a----          2023/6/9      7:22        7515594 result.mp4

```
使用 ffmpeg 合并视频和音频

```bash
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\results> ffmpeg -y -i .\concated_output.avi -i ..\inputs\demo_5_y.mp3  -map 0 -map 1:a -c:v copy -shortest  .\final_with_audio.avi
ffmpeg version git-2020-08-31-4a11a6f Copyright (c) 2000-2020 the FFmpeg developers
........
frame= 1793 fps=699 q=-1.0 Lsize=   25618kB time=00:00:59.76 bitrate=3511.2kbits/s speed=23.3x
video:24577kB audio:934kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.417315%
PS C:\Users\山河已无恙\Documents\GitHub\Wav2Lip-GFPGAN\results>
```

生成结果

[https://www.bilibili.com/video/BV1914y1U7dH/](https://www.bilibili.com/video/BV1914y1U7dH/)


关于 Demo 和小伙伴分享到这里
## 博文部分内容参考

© 文中涉及参考链接内容版权归原作者所有，如有侵权请告知，这是一个开源项目，如果你认可它，不要吝啬星星哦 :)


***

https://github.com/ajay-sainy/Wav2Lip-GFPGAN
***

© 2018-2023 liruilonger@gmail.com, All rights reserved. 保持署名-非商用-相同方式共享(CC BY-NC-SA 4.0)
.com, All rights reserved. 保持署名-非商用-相同方式共享(CC BY-NC-SA 4.0)
