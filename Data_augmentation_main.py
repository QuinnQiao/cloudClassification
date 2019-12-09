import os
import csv
from PIL import Image
import data_augmentation_fun
import matplotlib.pyplot as plt

path_lable = "./Train_label.csv"
path_data = "./Train/"
path_out = "./Train_new/"
path_new_lable = "./New_train_lable.csv"

name_lable = {}
new_train_lable = []
def load_lable():
	with open(path_lable, 'r', encoding = 'utf-8') as tmn:
		reader=csv.reader(tmn)
		for line in reader:
			name_lable[line[0]] = line[1]
	print("Get label completed")

def data_argumentation():
	img_augment = data_augmentation_fun.Rand_Augment()
	file_name_list = os.listdir(path_data)
	for file_name in file_name_list:
		if file_name in name_lable:
			label = name_lable[file_name]
		else:
			print("Error-no label in csv")
			continue
		src_img = Image.open(os.path.join(path_data,file_name))
		#Save raw img and lable
		src_img.save(os.path.join(path_out,"r"+file_name))
		new_train_lable.append(["r"+file_name, label])
		#First randaugmentation
		f_img = img_augment(src_img)
		#Save f_img and lable
		f_img.save(os.path.join(path_out, "f"+file_name))
		new_train_lable.append(["f"+file_name, label])
		#Second randaugmentation
		s_img = img_augment(src_img)
		#Save s_img and lable
		s_img.save(os.path.join(path_out, "s"+file_name))
		new_train_lable.append(["s"+file_name, label])
	print("Data argumentation completed")

def write_new_lable():
	with open(path_new_lable, 'w', newline='', encoding="utf-8") as csvfile:
		spamwriter = csv.writer(csvfile)
		spamwriter.writerow(["FileName","Code"])
		#csv_data [[orgList, line[0]]]
		for i in new_train_lable:
			writer = i
			spamwriter.writerow(writer)
		print("Write new lable completed")

if __name__ == "__main__":
	load_lable()
	data_argumentation()
	write_new_lable()
	print("Processing completed")
	






