
import numpy as np




def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# def handwritingClassTest():
#     hwLabels = []
#     trainingFileList = np.listdir('trainingDogits')
#     m = len(trainingFileList)
#     trainingMat = np.zeros((m, 1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         hwLabels.append(classNumStr)
#         trainingMat[i, :] = img2vector(fileNameStr)
#     testFileList = np.listdir('testDigits')
#     errorCount = 0
#     mTest = len(testFileList)
#     for i in range(mTest):
#         fileNameStr = testFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         vectorUnderTest = img2vector(fileNameStr)
#         classFileResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
#         if (classFileResult != classNumStr):
#             errorCount += 1







if __name__ == '__main__':
    test = img2vector('../datas/mlaction/Ch02/trainingDigits/0_13.txt')
    print(test)