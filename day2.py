#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   day2.py
@Time    :   2023/06/09 21:45:32
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   图片合成视频
"""

# here put the import lib


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
