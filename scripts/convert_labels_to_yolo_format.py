import sys, os
import cv2
import numpy as np
from tqdm import tqdm



print("Starting ..")


data_type = "train"
read_dir = "./visdrone/labels_raw/{0}/".format(data_type)
save_dir = "./visdrone/labels/{0}/".format(data_type)
save_format = "{0} {1} {2} {3} {4}"

to_coco = np.zeros(11, np.int32)
to_coco[0] = -1
to_coco[3] = 1
to_coco[4] = 2
to_coco[5] = 2
to_coco[6] = 7
to_coco[7] = 3
to_coco[8] = 3
to_coco[9] = 5
to_coco[10] = 3


for file in tqdm(os.listdir(read_dir)):

	if file.endswith(".txt"):

		filename = file.replace(".txt", "").strip()

		img = cv2.imread("./visdrone/images/" + data_type + "/" + filename + ".jpg")
		annotations = read_dir + filename + ".txt"

		img_h, img_w, _ = img.shape
		del img


		first_line = True

		detection_results = open(annotations, "r")
		data = detection_results.readline()

		

		while data:
			data_split = data.split(',')

			if data_split[4] == "0" or data_split[5] == "11" or data_split[-1] == "2\n":
				data = detection_results.readline()
				continue
			
			cls = to_coco[int(data_split[5])]

			box_ltwh = np.expand_dims(np.array(data_split[:4], dtype=np.float32), axis=0) # left, top, width, height
			box_xywh = np.copy(box_ltwh) # center_x, center_y, width, height

			box_xywh[0,0] = box_ltwh[0,0] + (0.5 * box_ltwh[0,2]) # center_x
			box_xywh[0,1] = box_ltwh[0,1] + (0.5 * box_ltwh[0,3]) # center_y
			
			box_xywh[0,0], box_xywh[0,2] = box_xywh[0,0] / img_w, box_xywh[0,2] / img_w
			box_xywh[0,1], box_xywh[0,3] = box_xywh[0,1] / img_h, box_xywh[0,3] / img_h

			with open(save_dir + file, 'a') as f:
				if not first_line:
					f.write("\n")
				else:
					first_line = False
				
				line = save_format.format(cls, box_xywh[0,0], box_xywh[0,1], box_xywh[0,2], box_xywh[0,3])
				f.write(line)

			data = detection_results.readline()
			

		

		detection_results.close()
	




print("Done.")
