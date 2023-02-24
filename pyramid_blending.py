

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
# Importing from other files
# from GetMask import GetMask
# from hybrid_images import gaussianKernel

# Read source, target and mask for a given id


def Read(id, path=""):
    source = plt.imread(path + "source_" + id + ".jpg")
    # get information about the image type (min max values)
    info = np.iinfo(source.dtype)
    # normalize the image into range 0 and 1
    source = source.astype(np.float32) / info.max
    target = plt.imread(path + "target_" + id + ".jpg")
    # get information about the image type (min max values)
    info = np.iinfo(target.dtype)
    # normalize the image into range 0 and 1
    target = target.astype(np.float32) / info.max
    mask = plt.imread(path + "mask_" + id + ".jpg")
    # get information about the image type (min max values)
    info = np.iinfo(mask.dtype)
    # normalize the image into range 0 and 1
    mask = mask.astype(np.float32) / info.max

    return source, mask, target

# Pyramid Blend


def PyramidBlend(source, mask, target, **kwargs):
    """_summary_

    Args:
        source (_type_): _description_
        mask (_type_): _description_
        target (_type_): _description_
        **kwargs (_type_): Used for overwriting various parameters like gaussian kernel

    Returns:
        _type_: _description_
    """

    # These defaults are used for calling the various sub functions
    defaults = {
        "Steps": 4
    }
    for key in kwargs:
        if key in defaults:
            defaults[key] = kwargs[key]

    # Generating the gaussian kernel for the mask
    # skimage.transform.resize #Already performs gaussian subsampling. Need to make sure anti_aliasing is true
    # For each gaussian pyramid step we are SMOOTHING THEN subsampling
    # Prior to this, we need to ensure that the images are of the correct size

    mergedLaplace = []  # Contains the merged laplacian output for each step

    pM = mask
    pSource = source
    pTarget = target
    for step in range(0, defaults["Steps"]):
        # Want odd numbers to be larger even numer
        dim = pM.shape
        newSize = np.ceil(np.asarray(dim)/2)
        newSize[-1] = dim[-1]

        # Creating the new images after resizing
        nM = resize(pM, newSize, anti_aliasing=True)
        nSource = resize(pSource, newSize, anti_aliasing=True)
        nTarget = resize(pTarget, newSize, anti_aliasing=True)

        # Upsampling again to determine the Laplacian difference
        upSource = resize(nSource, pSource.shape)
        upTarget = resize(nTarget, pTarget.shape)

        # Creating the difference per layer
        lSource = pSource - upSource
        lTarget = pTarget - upTarget
        print(pM.shape, lSource.shape, pSource.shape, "HERE")

        mergedLaplace.append((pM)*lSource+((1-pM)*lTarget))
        pM = nM
        pSource = nSource
        pTarget = nTarget

    # Getting the final, small image
    topImage = pM*pSource + (1-pM)*pTarget
    # putting the top image in the back to be scaled back up
    mergedLaplace.append(topImage)
    # Scaling back up
    # When we are scaling back up we will resize to the size given in the mergedLaplace
    mergedLaplace = mergedLaplace[::-1]
    for step in range(0, len(mergedLaplace)-1):
        scaledUp = resize(mergedLaplace[step], mergedLaplace[step+1].shape)
        mergedLaplace[step+1] = scaledUp+mergedLaplace[step+1]

    # Normalized to avoid color errors
    output = mergedLaplace[-1]-np.min(mergedLaplace[-1])
    output = output/np.max(output)
    return output


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image
    index = 3

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    # For the purposes of this lab ALL IMAGES ARE 300x300
    # Lazy way of enforcing this rule, takes bottom left of each image
    source = resize(source, mask.shape)
    target = resize(target, mask.shape)
    # mask = mask[:300, :300]
    # source = source[:300, :300]
    # target = target[:300, :300]

    ### The main part of the code ###
    # Implement the PyramidBlend function (Task 2)

    pyramidOutput = PyramidBlend(source, mask, target)

    # Writing the result

    plt.imsave("{}pyramid_{}.jpg".format(
        outputDir, str(index).zfill(2)), pyramidOutput)
