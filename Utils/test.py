def make_dataset(filepath, method):
    raw_dataset = tf.data.TFRecordDataset(filepath)
    
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }
    
    def preprocess_image(encoded_image):
        image = tf.image.decode_jpeg(encoded_image, 3)
        
        # resize to 286x286
        # image = tf.image.random_crop(image, [tf.shape(image)[0], tf.shape(image)[0], tf.shape(image)[-1]])
        image = tf.image.resize(image, [286, 286], method=method)
        
        # random crop a 256x256 area
        image = tf.image.random_crop(image, [256, 256, tf.shape(image)[-1]], seed=tf.random.set_seed(1))
        image = tf.cast(image, tf.float32)
  
        # normalize from 0-255 to -1 to +1
        image = tf.divide(image, 127.5) - 1.0

        return image
    
    def parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.io.parse_single_example(example_proto, image_feature_description)
        encoded_image = features['image/encoded']
        image = preprocess_image(encoded_image)
        return image
    
    parsed_image_dataset = raw_dataset.map(parse_image_function)
    return parsed_image_dataset


def generate_images(model, test_input, name, save=False):
    prediction = model(test_input)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.

        img = display_list[i]
        Ht = tf.shape(img)[0]
        Wt = tf.cast(tf.math.multiply(1.25, tf.cast(Ht, tf.float16)), tf.int32)
        c = tf.shape(img)[-1]
        img = tf.image.resize(img, [Ht, Wt], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if i==1:
            img = tf.image.adjust_jpeg_quality(img, jpeg_quality=100)

        plt.imshow(img * 0.5 + 0.5)
        plt.axis('off')
        if save:
            plt.savefig(fname=name, bbox_inches='tight')
        plt.show()
