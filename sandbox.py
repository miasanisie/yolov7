import sys
import time
import cv2
import numpy as np

from utils.general import segments2boxes, xyxy2xywh


print("Starting ..")


categories = [

]

# color = (255, 123, 37)
colors = [
	(255, 128, 0),
	(255, 153, 153),
	(255, 255, 0),
	(0, 255, 0),
	(0, 255, 255),
	(0, 191, 255),
	(0, 0, 255),
	(128, 0, 255),
	(255, 0, 255),
	(255, 0, 0)
]

coco_names = np.load("./coco_names.npy")

"""
file = "000000000139"

img = cv2.imread("./coco/images/val2017/" + file + ".jpg")
annotations = "./coco/labels/val2017/" + file + ".txt"

img_h, img_w, _ = img.shape
print(img.shape)

detection_results = open(annotations, "r")
data = detection_results.readline()

while data:
	data_split = data.strip().split(' ')

	print(data_split)
	segments = [np.array(data_split[1:], dtype=np.float32).reshape(-1, 2)]
	# print(segments)
	boxes = segments2boxes(segments).squeeze()
	print(boxes)


	x_c = boxes[0] * img_w
	y_c = boxes[1] * img_h
	w = boxes[2] * img_w
	h = boxes[3] * img_h

	img = cv2.rectangle(img, (int(x_c - (0.5 * w)), int(y_c - (0.5 * h))), (int(x_c + (0.5 * w)), int(y_c + (0.5 * h))), colors[0], 2)

	# break

	data = detection_results.readline()



detection_results.close()


cv2.imwrite("./testing.jpg", img)


sys.exit()
# """



file = "9999955_00000_d_0000110"

img = cv2.imread("./visdrone/images/train/" + file + ".jpg")
annotations = "./visdrone/labels/train/" + file + ".txt"

img_h, img_w, _ = img.shape
print(img.shape)

detection_results = open(annotations, "r")
data = detection_results.readline()

while data:
	data_split = data.split(' ')
	box_ltwh = np.expand_dims(np.array(data_split[:4], dtype=np.float32), axis=0) # left, top, width, height
	box_xywh = np.expand_dims(np.array(data_split[1:], dtype=np.float32), axis=0) # np.copy(box_ltwh)
	"""
	box_xywh[0,0] = box_ltwh[0,0] + (0.5 * box_ltwh[0,2]) # x
	box_xywh[0,1] = box_ltwh[0,1] + (0.5 * box_ltwh[0,3]) # y
	
	print(box_ltwh)
	print(box_xywh)

	box_xywh[0,0], box_xywh[0,2] = box_xywh[0,0] / img_w, box_xywh[0,2] / img_w
	box_xywh[0,1], box_xywh[0,3] = box_xywh[0,1] / img_h, box_xywh[0,3] / img_h
	
	print(box_xywh)
	# """
	x_c = box_xywh[0,0] * img_w
	y_c = box_xywh[0,1] * img_h
	w = box_xywh[0,2] * img_w
	h = box_xywh[0,3] * img_h

	img = cv2.rectangle(img, (int(x_c - (0.5 * w)), int(y_c - (0.5 * h))), (int(x_c + (0.5 * w)), int(y_c + (0.5 * h))), colors[0], 2)
	print("{0}: {1}".format(data_split[0], coco_names[int(data_split[0])]))
	cv2.imwrite("./testing.jpg", img)
	time.sleep(3)
	# break

	# print(data_split)

	# img = cv2.rectangle(img, (int(data_split[0]), int(data_split[1])), (int(data_split[0]) + int(data_split[2]), int(data_split[1]) + int(data_split[3])), colors[int(data_split[5]) - 1], 2)

	data = detection_results.readline()



detection_results.close()


# cv2.imwrite("./testing.jpg", img)
# cv2.imwrite("./class__motor.jpg", img)


########################################################

# python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/visdrone.yaml --img 1280 1280 --cfg cfg/training/yolov7-d6.yaml --weights 'yolov7-d6_training.pt' --name yolov7-d6-visdrone --hyp data/hyp.scratch.custom.yaml




print("Done.")
