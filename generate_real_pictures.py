import matplotlib.pyplot as plt
import os
import glob

INPUT_DIR = "Results/"

def main():
	folders = ["MonetToLandscapeInput", "MonetToLandscape"]
	
	# plt.figure(figsize=(13, 25))
	# type = 0
	# fsize = 30
	# for file in folders:
	# 	test_images = sorted(glob.glob(os.path.join(INPUT_DIR, file, '*.jpg')))
	# 	type += 1
	# 	counter = 0
	# 	for img_path in test_images[:6]:
	# 		img = plt.imread(img_path)
	# 		idx = 2 * counter + type
	# 		plt.subplot(6, 2, idx)
	# 		plt.imshow(img)
	# 		if idx == 1:
	# 			plt.title("Input", fontsize=fsize)
	# 		elif idx == 2:
	# 			plt.title("Output", fontsize=fsize)
	# 		plt.axis('off')
	# 		counter += 1
	#
	# plt.savefig(fname="MonetToLandscape1.png", bbox_inches='tight')
	# plt.show()
	# print('Saved Images 1')
	#
	# plt.figure(figsize=(13, 25))
	# type = 0
	# for file in folders:
	# 	test_images = sorted(glob.glob(os.path.join(INPUT_DIR, file, '*.jpg')))
	# 	type += 1
	# 	counter = 0
	# 	for img_path in test_images[6:12]:
	# 		img = plt.imread(img_path)
	# 		idx = 2 * counter + type
	# 		plt.subplot(6, 2, idx)
	# 		plt.imshow(img)
	# 		if idx == 1:
	# 			plt.title("Input", fontsize=fsize)
	# 		elif idx == 2:
	# 			plt.title("Output", fontsize=fsize)
	# 		plt.axis('off')
	# 		counter += 1
	#
	# plt.savefig(fname="MonetToLandscape2.png", bbox_inches='tight')
	# plt.show()
	# print('Saved Images 2')
	
	plt.figure(figsize=(27, 25))
	counter = 0
	folders = ["MonetToLandscape1.png", "MonetToLandscape2.png"]
	for img_path in folders:
		img = plt.imread(img_path)
		counter += 1
		plt.subplot(1, 2, counter)
		plt.imshow(img)
		plt.axis('off')
	
	plt.savefig(fname="MonetToLandscape.png", bbox_inches='tight')
	plt.show()
	print('Saved Images')
	
	
if __name__ == '__main__':
	main()
