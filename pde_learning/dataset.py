################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random

# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = True
# global switch, make data dimensionless?
makeDimLess = True
# global switch, remove constant offsets from pressure channel?
removePOffset = True

## helper - compute absolute of inputs or targets
def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval


######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #

def LoaderNormalizer(data, isTest = False, shuffle = 0, dataProp = None):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """

    if dataProp is None:
        # load single directory
        files = listdir(data.dataDir)
        files.sort()
        for i in range(shuffle):
            random.shuffle(files) 
        if isTest:
            print("Reducing data to load for tests")
            files = files[0:min(10, len(files))]
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))

        for i, file in enumerate(files):
            npfile = np.load(data.dataDir + file)
            d = npfile['a']
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]
        print("Number of data loaded:", len(data.inputs) )

    else:
        # load from folders reg, sup, and shear under the folder dataDir
        data.totalLength = int(dataProp[0])
        data.inputs  = np.empty((data.totalLength, 3, 128, 128))
        data.targets = np.empty((data.totalLength, 3, 128, 128))

        files1 = listdir(data.dataDir + "reg/")
        files1.sort()
        files2 = listdir(data.dataDir + "sup/")
        files2.sort()
        files3 = listdir(data.dataDir + "shear/" )
        files3.sort()
        for i in range(shuffle):
            random.shuffle(files1) 
            random.shuffle(files2) 
            random.shuffle(files3) 

        temp_1, temp_2 = 0, 0
        for i in range(data.totalLength):
            if i >= (1-dataProp[3])*dataProp[0]:
                npfile = np.load(data.dataDir + "shear/" + files3[i-temp_2])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
            elif i >= (dataProp[1])*dataProp[0]:
                npfile = np.load(data.dataDir + "sup/" + files2[i-temp_1])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
                temp_2 = i + 1
            else:
                npfile = np.load(data.dataDir + "reg/" + files1[i])
                d = npfile['a']
                data.inputs[i] = d[0:3]
                data.targets[i] = d[3:6]
                temp_1 = i + 1
                temp_2 = i + 1
        print("Number of data loaded (reg, sup, shear):", temp_1, temp_2 - temp_1, i+1 - temp_2)

    ################################## NORMALIZATION OF TRAINING DATA ##########################################

    if removePOffset:
        for i in range(data.totalLength):
            data.targets[i,0,:,:] -= np.mean(data.targets[i,0,:,:]) # remove offset
            data.targets[i,0,:,:] -= data.targets[i,0,:,:] * data.inputs[i,2,:,:]  # pressure * mask

    # make dimensionless based on current data set
    if makeDimLess:
        for i in range(data.totalLength):
            # only scale outputs, inputs are scaled by max only
            v_norm = ( np.max(np.abs(data.inputs[i,0,:,:]))**2 + np.max(np.abs(data.inputs[i,1,:,:]))**2 )**0.5 
            data.targets[i,0,:,:] /= v_norm**2
            data.targets[i,1,:,:] /= v_norm
            data.targets[i,2,:,:] /= v_norm

    # normalize to -1..1 range, from min/max of predefined
    if fixedAirfoilNormalization:
        # hard coded maxima , inputs dont change
        data.max_inputs_0 = 100.
        data.max_inputs_1 = 38.12
        data.max_inputs_2 = 1.0

        # targets depend on normalization
        if makeDimLess:
            data.max_targets_0 = 4.65 
            data.max_targets_1 = 2.04
            data.max_targets_2 = 2.37
            print("Using fixed maxima "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] ))
        else: # full range
            data.max_targets_0 = 40000.
            data.max_targets_1 = 200.
            data.max_targets_2 = 216.
            print("Using fixed maxima "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] ))

    else: # use current max values from loaded data
        data.max_inputs_0 = find_absmax(data, 0, 0)
        data.max_inputs_1 = find_absmax(data, 0, 1)
        data.max_inputs_2 = find_absmax(data, 0, 2) # mask, not really necessary
        print("Maxima inputs "+format( [data.max_inputs_0,data.max_inputs_1,data.max_inputs_2] )) 

        data.max_targets_0 = find_absmax(data, 1, 0)
        data.max_targets_1 = find_absmax(data, 1, 1)
        data.max_targets_2 = find_absmax(data, 1, 2)
        print("Maxima targets "+format( [data.max_targets_0,data.max_targets_1,data.max_targets_2] )) 

    data.inputs[:,0,:,:] *= (1.0/data.max_inputs_0)
    data.inputs[:,1,:,:] *= (1.0/data.max_inputs_1)

    data.targets[:,0,:,:] *= (1.0/data.max_targets_0)
    data.targets[:,1,:,:] *= (1.0/data.max_targets_1)
    data.targets[:,2,:,:] *= (1.0/data.max_targets_2)

    ###################################### NORMALIZATION  OF TEST DATA #############################################

    if isTest:
        files = listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 3, 128, 128))
        data.targets = np.empty((len(files), 3, 128, 128))
        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            d = npfile['a']
            data.inputs[i] = d[0:3]
            data.targets[i] = d[3:6]

        if removePOffset:
            for i in range(data.totalLength):
                data.targets[i,0,:,:] -= np.mean(data.targets[i,0,:,:]) # remove offset
                data.targets[i,0,:,:] -= data.targets[i,0,:,:] * data.inputs[i,2,:,:]  # pressure * mask

        if makeDimLess:
            for i in range(len(files)):
                v_norm = ( np.max(np.abs(data.inputs[i,0,:,:]))**2 + np.max(np.abs(data.inputs[i,1,:,:]))**2 )**0.5 
                data.targets[i,0,:,:] /= v_norm**2
                data.targets[i,1,:,:] /= v_norm
                data.targets[i,2,:,:] /= v_norm
    
        data.inputs[:,0,:,:] *= (1.0/data.max_inputs_0)
        data.inputs[:,1,:,:] *= (1.0/data.max_inputs_1)

        data.targets[:,0,:,:] *= (1.0/data.max_targets_0)
        data.targets[:,1,:,:] *= (1.0/data.max_targets_1)
        data.targets[:,2,:,:] *= (1.0/data.max_targets_2)

    print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % ( 
      np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False) , 
      np.mean(np.abs(data.inputs), keepdims=False) , np.max(np.abs(data.inputs), keepdims=False) ) ) 

    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode==self.TEST), dataProp=dataProp, shuffle=shuffle)

        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 400)

            self.valiInputs = self.inputs[targetLength:]
            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]
            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    #  reverts normalization 
    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] /= (1.0/self.max_targets_0)
        a[1,:,:] /= (1.0/self.max_targets_1)
        a[2,:,:] /= (1.0/self.max_targets_2)

        if makeDimLess:
            a[0,:,:] *= v_norm**2
            a[1,:,:] *= v_norm
            a[2,:,:] *= v_norm
        return a

# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

