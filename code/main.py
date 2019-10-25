# import external libraries
import sys, os, re
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
# import my files
from models import HMM, Brill, StanfordModel

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def prepareNLTK(directory):
    tagList = []
    sentenceList = []
    with open(directory, "r") as file:
        for line in file:
            split = line.split(" ")
            if len(split) > 1:
                sentenceList.append((split[0], split[1][:-1])) # delete new line characters
            else:
                tagList.append(sentenceList)
                sentenceList.clear()

    return tagList

def main(mode, directory):
    if mode == "--stanford":
        # prep files
        replace(directory, " ", "/")
        testDirectory = re.sub("Train", "Test", directory)
        replace(testDirectory, " ", "/")
        fileName = "Stanford1.tagger" if "Domain1" in directory else "Stanford2.tagger"
        model = StanfordModel(fileName, directory)
        model.train()
        model.test(testDirectory)
        results = open(fileName, "w+")

    elif mode == "--nltk":
        # prep files
        replace(directory, "/", " ")
        testDirectory = re.sub("Train", "Test", directory)
        replace(testDirectory, "/", " ")

        tagList = prepareNLTK(directory)
        testList = prepareNLTK(testDirectory)

        fileName = "nltk1.txt" if "Domain1" in directory else "nltk2.txt"

        modelHMM = HMM(text=tagList, directory=directory, testText=testList)
        trainedHMM = modelHMM.train()
        modelBrill = Brill(text=tagList, directory=directory, testText=testList, tagger=trainedHMM)
        modelBrill.train()

        hmmAcc = modelHMM.test()
        brillAcc = modelBrill.test()

        with open(fileName, "w+") as file:
            file.write("HMM accuracy: " + str(hmmAcc) + "\n")
            file.write("Brill accuracy: " + str(brillAcc) + "\n")


    else:
        print("Incorrect usage: mode must be either --stanford or --nltk")



if __name__ == '__main__':
    #check command line arguments
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Usage: ./main.py <--stanford or --nltk> <.txt file to model>\n")