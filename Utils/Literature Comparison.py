import matplotlib.pyplot as plt
import tensorflow as tf

INPUT_DIR = "Results/Literature Comparison/"


def preprocess_image(image_path):
	image = plt.imread(image_path)
	
	# resize to 286x286
	Ht = tf.shape(image)[0]
	Wt = tf.cast(tf.math.multiply(1.5, tf.cast(Ht, tf.float16)), tf.int32)
	image = tf.image.resize(image, [Ht, Wt], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	return image


def main():
	
	plt.figure(figsize=(45, 15))
	fsize =45
	img_path = INPUT_DIR
	
	# Chicago
	img = preprocess_image(img_path + "chicago.jpg")
	plt.subplot(2, 5, 1)
	plt.imshow(img)
	plt.title("Input", fontsize=fsize)
	plt.axis('off')
	
	# Style Image
	img = preprocess_image(img_path + "Starry_night.jpg")
	plt.subplot(2, 5, 2)
	plt.imshow(img)
	plt.title("Starry Night", fontsize=fsize)
	plt.axis('off')
	
	
	# Gatys Starry
	img = preprocess_image(img_path + "Gatys_Starry.jpg")
	plt.subplot(2, 5, 3)
	plt.imshow(img)
	plt.title("Gatys et al.", fontsize=fsize)
	plt.axis('off')
	
	
	# Justin Starry
	img = preprocess_image(img_path + "Justin_Starry.jpg")
	plt.subplot(2, 5, 4)
	plt.imshow(img)
	plt.title("Johnson et al.", fontsize=fsize)
	plt.axis('off')
	
	# CycleGAN Starry
	img = preprocess_image(img_path + "CycleGAN_Starry.jpg")
	plt.subplot(2, 5, 5)
	plt.imshow(img)
	plt.title("CycleGAN", fontsize=fsize)
	plt.axis('off')
	
	# Chicago
	img = preprocess_image(img_path + "chicago.jpg")
	plt.subplot(2, 5, 6)
	plt.imshow(img)
	plt.title("Input", fontsize=fsize)
	plt.axis('off')
	
	# Style Image
	img = preprocess_image(img_path + "Wave.jpg")
	plt.subplot(2, 5, 7)
	plt.imshow(img)
	plt.title("Gatys et al.", fontsize=fsize)
	plt.axis('off')
	
	# Gatys Starry
	img = preprocess_image(img_path + "Gatys_Wave.jpg")
	plt.subplot(2, 5, 8)
	plt.imshow(img)
	plt.title("Gatys et al.", fontsize=fsize)
	plt.axis('off')
	
	# Justin Starry
	img = preprocess_image(img_path + "Justin_Wave.jpg")
	plt.subplot(2, 5, 9)
	plt.imshow(img)
	plt.title("Johnson et al.", fontsize=fsize)
	plt.axis('off')
	
	# CycleGAN Starry
	img = preprocess_image(img_path + "CycleGAN_Wave.jpg")
	plt.subplot(2, 5, 10)
	plt.imshow(img)
	plt.title("CycleGAN", fontsize=fsize)
	plt.axis('off')
	
	# plt.savefig(fname="Artistic_Style3.png", bbox_inches='tight')
	plt.show()
	print('Saved Images')


if __name__ == '__main__':
	main()
