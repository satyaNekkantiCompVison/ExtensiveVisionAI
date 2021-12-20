import os
import random

filenames = os.listdir("/home/duranc/EVA7/ExtensiveVisionAI/S11_YOLO_v1_v2_v3/CustomYoloTrain/data/customData/images")


sample_structure = "./data/customData/images/"
random.shuffle(filenames)

idx = 0

file1 = open("train.txt", "w")  # append mode
while idx<90:
	train_image = sample_structure+filenames[idx]
	file1.write(train_image+"\n")
	idx+=1

file1.close()

idx=90
file2 = open("test.txt","w")
while idx<100:
	test_image = sample_structure+filenames[idx]
	file2.write(test_image+"\n")
	idx+=1

file2.close()
