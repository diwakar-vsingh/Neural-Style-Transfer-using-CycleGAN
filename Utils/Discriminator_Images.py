import matplotlib.pyplot as plt
import os
import glob

INPUT_DIR = "Results/Discriminator"


def main():
	folders = ["Input", "PixelGAN", "PatchGAN1", "PatchGAN2", "PatchGAN3"]
	
	plt.figure(figsize=(20, 15))
	type = 0
	fsize = 35
	for file in folders:
		test_images = sorted(glob.glob(os.path.join(INPUT_DIR, file, '*.jpg')))
		type += 1
		counter = 0
		for img_path in test_images[0:5]:
			img = plt.imread(img_path)
			idx = 5 * counter + type
			plt.subplot(5, 5, idx)
			plt.imshow(img)
			if idx == 1:
				plt.title("Input", fontsize=fsize)
			elif idx == 2:
				plt.title("1x1", fontsize=fsize)
			elif idx == 3:
				plt.title("16x16", fontsize=fsize)
			elif idx == 4:
				plt.title("70x70", fontsize=fsize)
			elif idx == 5:
				plt.title("256x256", fontsize=fsize)
			plt.axis('off')
			counter += 1
	# plt.savefig(fname="Artistic_Style3.png", bbox_inches='tight')
	plt.show()
	print('Saved Images')


if __name__ == '__main__':
	main()
