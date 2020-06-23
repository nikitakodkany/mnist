import numpy as np

def load_data(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix +'-images-idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + '-labels-idx1-ubyte', dtype ='ubyte')[2 * intType.itemsize:]

    return data, labels

def toReturn():
    traininImages, trainingLabels = load_data("train", "dataset")
    testImages, testLabels = load_data("t10k", "dataset")
    print()
    print("loading Images...")
    print("loading Labels...")
    print()

    return traininImages, trainingLabels, testImages, testLabels
