#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   day1.py
@Time    :   2023/06/09 21:44:59
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   视频切片为图片
"""

# here put the import lib


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



