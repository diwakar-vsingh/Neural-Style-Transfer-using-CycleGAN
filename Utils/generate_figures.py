import matplotlib.pyplot as plt
import os
import glob

OUTPUT_DIR = "Results/"
INPUT_DIR = "Results/"

def main():
	folders = ["Input", "MonetPaintings", "CezannePaintings", "Ukiyo_ePaintings", "VanGoghPaintings"]
	
	plt.figure(figsize=(20, 31))
	type = 0
	fsize = 40
	for file in folders:
		test_images = sorted(glob.glob(os.path.join(INPUT_DIR, file, '*.jpg')))
		type += 1
		counter = 0
		for img_path in test_images[20:30]:
			img = plt.imread(img_path)
			idx = 5*counter + type
			plt.subplot(10, 5, idx)
			plt.imshow(img)
			if idx == 1:
				plt.title("Input", fontsize=fsize)
			elif idx == 2:
				plt.title("Monet", fontsize=fsize)
			elif idx == 3:
				plt.title("Cezanne", fontsize=fsize)
			elif idx == 4:
				plt.title("Ukiyo-e", fontsize=fsize)
			elif idx == 5:
				plt.title("Van Gogh", fontsize=fsize)
			plt.axis('off')
			counter += 1
	plt.savefig(fname="Artistic_Style3.png", bbox_inches='tight')
	plt.show()
	print('Saved Images')


if __name__ == '__main__':
	main()
