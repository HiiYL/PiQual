# initialize the total number of files 
# and input file list
numberOfFiles=0
inputFileList=[]
hdfFileList=[]
channel=3
visualize=False
# open train or test file list with label
with open(inputFile, 'r') as inputData:
    for fileName in inputData: 
        # this input file list includes label information as well
        inputFileList.append(fileName)  
        numberOfFiles = numberOfFiles + 1 

print "A number of files: ", numberOfFiles

# this loop will open file from inputFileList one by one and put it into
# hdf output

# initialize index 
index=0
fileIndex=0
periodNum=100

if numberOfFiles < periodNum:
    periodNum=numberOfFiles


# loop through thed list of data filess
for dataFileName in inputFileList:

    if (fileIndex % periodNum) == 0:

        # open and create hdf5 file output directory
        outputHDFFile = fileType + "-" + str(fileIndex) + ".h5"
        print "file name: " + outputHDFFile
        outputHDFPath = join(outputDir, outputHDFFile)
        print "hdf5 file: ", outputHDFPath
        fileOut = h5py.File(outputHDFPath, 'w')
        hdfFileList.append(outputHDFPath)

        data = fileOut.create_dataset("data", (periodNum,channel,width,height), dtype=np.float32)
        label = fileOut.create_dataset("label", (periodNum,), dtype=np.float32)

        # image data matrix
        imageStack = np.empty((periodNum,channel,width,height)) # Create empty HxWxN array/matrix
        labelStack = np.empty((periodNum))
        # initialize index at every periodNum 
        index=0

    dataPathandLabel=dataFileName.split(' ', 1)
    dataFilePath=dataPathandLabel[0]
    # print(dataFilePath)
    dataLabel=dataPathandLabel[1]
    # print(dataLabel)
    lastSubDirName=dataFilePath.split('/')
    subDirName=lastSubDirName[-1]
    # print(subDirName)

    labelNumber=int(dataLabel)
    # print labelNumber

    # load image:
    if channel == 1: 
        img=cv2.imread(dataFilePath, cv2.CV_LOAD_IMAGE_GRAYSCALE) # load grayscale
        print 'grayscale: ', img.shape

        imageStack[index,:,:,:]=img
        labelStack[index]=labelNumber

    elif channel == 3:
        img = cv2.imread(dataFilePath, cv2.CV_LOAD_IMAGE_COLOR) # color
        # height, width = img.shape[:2]
        img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
        if index < 5 and visualize:
            plt.imshow(img)
            plt.show()

        img = img.transpose(2,1,0)

        # print 'RGB', img.shape
        # imageStack[index,:,:,:]=img
        # labelStack[index]=labelNumber
        data[index,:,:,:]=img
        label[index]=labelNumber

    index=index+1
    fileIndex=fileIndex+1

    if (fileIndex % periodNum) == 0:
        # close the last file
        imageStack.__init__()
        labelStack.__init__()
        fileOut.close()
        print 'file close'


# list hdf5 train dataset file list
outputHDFListFile = fileType + '.txt'
outputHDFListPath = join(outputDir, outputHDFListFile)

if exists(outputHDFListPath): 
    outputHDFListFile = fileType + '-list.txt'
    outputHDFListPath = join(outputDir, outputHDFListFile)

print 'list: ', outputHDFListFile
print 'Output dir: ', outputHDFListPath

with open(outputHDFListPath, 'w') as trainOut:
    for hdfFile in hdfFileList:
        print hdfFile
        writeOut=hdfFile + "\n"
        trainOut.write(writeOut)