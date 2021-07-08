import os
import random
import numpy as np
from skimage.draw import line
from sklearn.model_selection import train_test_split
from imageio import imread
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
from skimage import color


def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:    
        im = im.astype(np.float64) / 255.0
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    image_paths = {}
    h, w = crop_size
    
    while True:
        batch = np.zeros((batch_size, h, w, 1, 2))
        
        for k in range(batch_size):
            path = random.choice(filenames)
            im = image_paths[path] = \
                image_paths.get(path, read_image(path, 1))  # cache image path
            
            # initially take 3 * crop_size sized patches
            i, j = np.random.randint(0, im.shape[0] - 3 * h + 1), \
                   np.random.randint(0, im.shape[1] - 3 * w + 1)
            patch = im[i:i + 3 * h, j:j + 3 * w]
            corrupted_patch = corruption_func(patch)
            
            # take crop_size sized patches
            i, j = np.random.randint(0, 2 * h + 1), \
                   np.random.randint(0, 2 * w + 1)
            
            patch = patch[i:i + h, j:j + w].reshape(h, w, 1)
            corrupted_patch = corrupted_patch[i:i + h, j:j + w].reshape(h, w, 1)
            
            # insert patches into batch
            batch[k, :, :, :, 0], batch[k, :, :, :, 1] = corrupted_patch - 0.5, patch - 0.5
        
        yield batch[:, :, :, :, 0], batch[:, :, :, :, 1]


def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    b = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    b = Activation('relu')(b)
    b = Conv2D(num_channels, (3, 3), padding='same')(b)
    b = Add()([input_tensor, b])
    return b


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    b = Activation('relu')(b)
    for i in range(num_res_blocks):
        b = resblock(b, num_channels)
    b = Conv2D(1, (3, 3), padding='same')(b)
    b = Add()([a, b])
    model = Model(inputs=a, outputs=b)
    return model


"""# 5 Training Networks for Image Restoration"""


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    # split data into train set and validation set
    train_set, valid_set = train_test_split(images, train_size=0.8, test_size=0.2)
    
    crop_size = model.input_shape[1], model.input_shape[2]
    train_generator = load_dataset(train_set, batch_size, corruption_func, crop_size)
    valid_generator = load_dataset(valid_set, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    
    # train the model with the above batches of image data
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=valid_generator, validation_steps=num_valid_samples // batch_size,
                        use_multiprocessing=True)


def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    im = corrupted_image - 0.5
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    im = im[np.newaxis, :, :, np.newaxis]
    im = new_model.predict(im)
    im = im + 0.5
    im = np.squeeze(np.clip(im, 0.0, 1.0))
    return im.astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sigma = np.random.uniform(low=min_sigma, high=max_sigma, size=1)[0]
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    corrupted = image + noise
    # round image values to nearest fracture of form (i / 255)
    corrupted = np.around(255*corrupted) / 255
    return np.clip(corrupted, 0.0, 1.0)  # clip the image to the range of [0, 1]


def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    def corruption_func(im):
        return add_gaussian_noise(im, 0, 0.2)
    
    images = images_for_denoising()
    model = build_nn_model(24, 24, 48, denoise_num_res_blocks)
    batch, steps, epochs, num_samples = (10, 3, 2, 30) if quick_mode\
        else (100, 100, 10, 1000)
    
    train_model(model, images, corruption_func,
                batch, steps, epochs, num_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    blurred_im = convolve(image, kernel)
    return blurred_im


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle = np.random.uniform(low=0, high=np.pi, size=1)[0]
    kernel_size = np.random.choice(list_of_kernel_sizes)
    blurred_im = add_motion_blur(image, kernel_size, angle)
    # round image values to nearest fracture of form (i / 255)
    blurred_im = np.around(255*blurred_im) / 255
    return np.clip(blurred_im, 0.0, 1.0)  # clip the image to the range of [0, 1]


def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    def corruption_func(im):
        return random_motion_blur(im, [7])

    images = images_for_deblurring()
    model = build_nn_model(16, 16, 32, deblur_num_res_blocks)
    batch, steps, epochs, num_samples = (10, 3, 2, 30) if quick_mode \
        else (100, 100, 10, 1000)

    train_model(model, images, corruption_func,
                batch, steps, epochs, num_samples)
    return model


def super_resolution_corruption(image):
    """
    Perform the super resolution corruption 
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    factor = np.random.choice([2, 3, 4])
    height, width = image.shape[0], image.shape[1]
    image = image[:(height // factor) * factor, :(width // factor) * factor]
    image = zoom(image, 1/factor)
    image = zoom(image, factor)
    return image


def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    def corruption_func(im):
        return super_resolution_corruption(im)

    images = images_for_super_resolution()
    model = build_nn_model(16, 16, 32, super_resolution_num_res_blocks)
    batch, steps, epochs, num_samples = (10, 3, 2, 30) if quick_mode \
        else (100, 100, 10, 1000)

    train_model(model, images, corruption_func,
                batch, steps, epochs, num_samples)
    return model
