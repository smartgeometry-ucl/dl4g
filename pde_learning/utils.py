################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helper functions for image output
#
################

import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm

# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint: print(line)

# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()

# compute learning rate with decay in second half
def computeLR(i,epochs, minLR, maxLR):
    if i < epochs*0.5:
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR-minLR)*f

# image output
def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    
    s = outputs.shape[1] # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ( (s+10)*3, s*2) , color=(255,255,255) )
        BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else: # from -1,1 to 0,1
            outputs[i] -= -1.
            targets[i] -= -1.
            outputs[i] /= 2.
            targets[i] /= 2.

        if not saveMontage:
            suffix = ""
            if i==0:
                suffix = "_pressure"
            elif i==1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512,512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*1))

            im = Image.fromarray(targets[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(outputs[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*1))
            imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            BW_im.paste(imE, ( (s+10)*i, s*2))

    if saveMontage:
        new_im.save(filename + ".png")
        BW_im.save( filename + "_bw.png")

# save single image
def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)

# read data split from command line
def readProportions():
    flag = True
    while flag:
        input_proportions = input("Enter total numer for training files and proportions for training (normal, superimposed, sheared respectively) seperated by a comma such that they add up to 1: ")
        input_p = input_proportions.split(",")
        prop = [ float(x) for x in input_p ]
        if prop[1] + prop[2] + prop[3] == 1:
            flag = False
        else:
            print( "Error: poportions don't sum to 1")
            print("##################################")
    return(prop)

# helper from data/utils
def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


