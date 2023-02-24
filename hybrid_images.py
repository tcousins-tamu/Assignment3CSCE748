"""
Credit: Alyosha Efros
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr

isGray = True


def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2*r+1 - R))
    cpad = (int)(np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)): -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)): -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)): -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)): -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def combine(im1, im2):
    """Combines two images using simple addition then normalizes them

    Args:
        im1 (_type_): array corresponding to image 1
        im2 (_type_): array corresponding to image 2

    Returns:
        _type_: array containing the finalized output
    """
    im = im1 + im2
    im = im / im.max()
    return im


def gaussianKernel(stdv=.1, mu=0, kernelSz=3):
    """Returns a gaussian kernel according to the two parameter gaussiam
    function 1/(2pi*sigma**2) * e**((-x**2+y**2)/(2*sigma**2))

    Args:
        stdv (float, optional): _description_. Defaults to .1.
        mu (float, optional): _description_. Defaults to 1.
        kernelSz (int, optional): _description_. Defaults to 3.
    Returns:
        array: array containing the Gaussian Kernel
    """
    # creating the kernel and initializing w/ garbage values
    x, y = np.meshgrid(np.linspace(-1, 1, kernelSz),
                       np.linspace(-1, 1, kernelSz))

    mag = np.sqrt(x**2+y**2)
    kernel = np.exp(-((mag-mu)**2 / (2.0 * stdv**2)))

    # need to normalize kernel
    return kernel/(np.max(kernel))

def hybridImages(im1, im2, stdv=.1, mu=1, kernelSz=3):
    """This Function Creates a Hybrid Image output given two input images.
    It does this by convolving a gaussian kernel (for a low pass filter) over 
    image 1 and the impulse minus the gaussian over image 2, then combining.

    Args:
        im1 (_type_): Low frequency Image
        im2 (_type_): High Frequency Image
        stdv (float, optional): Standard Deviation of the Gaussian filters. Defaults to .1.
        kernelSz (int, optional): Kernel Size for the gaussian filters. Defaults to 3.
    """

    # Step One: Forming the kernel for the low and high pass cases
    lowPass = gaussianKernel(stdv, mu, kernelSz)
    highPass = 1-lowPass

    lowIm1 = scipy.signal.convolve2d(im1, lowPass)
    highIm2 = scipy.signal.convolve2d(im2, highPass)

    return combine(lowIm1, highIm2)


if __name__ == "__main__":

    imageDir = './Images/'
    outDir = './Results/'

    im1_name = 'Monroe.jpg'
    im2_name = 'Einstein.jpg'

    # 1. load the images

    # Low frequency image
    im1 = plt.imread(imageDir + im1_name)  # read the input image
    # get information about the image type (min max values)
    info = np.iinfo(im1.dtype)
    # normalize the image into range 0 and 1
    im1 = im1.astype(np.float32) / info.max

    # High frequency image
    im2 = plt.imread(imageDir + im2_name)  # read the input image
    # get information about the image type (min max values)
    info = np.iinfo(im2.dtype)
    # normalize the image into range 0 and 1
    im2 = im2.astype(np.float32) / info.max

    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)

    if isGray:
        im1_aligned = np.mean(im1_aligned, axis=2)
        im2_aligned = np.mean(im2_aligned, axis=2)

    # Now you are ready to write your own code for creating hybrid images!
    im_low = im1_aligned
    im_high = im2_aligned

    #im = combine(im_low, im_high)
    im = hybridImages(im_low, im_high, stdv=.5, mu=0, kernelSz=3)
    print("here")
    if isGray:
        plt.imsave(outDir + im1_name[:-4] + '_' +
                   im2_name[:-4] + '_Hybrid.jpg', im, cmap='gray')
    else:
        plt.imsave(outDir + im1_name[:-4] + '_' +
                   im2_name[:-4] + '_Hybrid.jpg', im)
