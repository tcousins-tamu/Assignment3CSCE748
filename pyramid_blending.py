

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Pyramid Blend
def PyramidBlend(source, mask, target):
    
    return source * mask + target * (1 - mask)


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = './Images/'
    outputDir = './Results/'

    # main area to specify files and display blended image

    index = 1

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    
    ### The main part of the code ###

    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target)
    

    
    # Writing the result

    plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
