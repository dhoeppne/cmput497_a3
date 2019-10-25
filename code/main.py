# import external libraries
import sys, os, re
from tempfile import mkstemp
from shutil import move
import subprocess
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
                sentenceList = []

    return tagList

def testDirectory(model, files):
    output = []
    for testFile in files:
        o = model.test(testFile)
        if o is not None:
            output.append(o)

    return output

def main(mode, directory):
    precheck = os.listdir(".")
    for file in precheck:
        if "st_temp_" in file:
            bash_script = "rm -rf " + file
            process = subprocess.Popen(bash_script.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

    preFiles = os.listdir(directory)
    files = []
    for file in preFiles:
        if ".txt" in file:
            files.append(file)

    try:
        remove("./results.txt")
        f = open("results.txt", "w+")
        f.close()
    except:
        print("Could not create results file")

    for file in files:
        for wrong in files:
            # prep file based on mode
            if mode == "--stanford":
                replace(os.path.join(directory, wrong), " ", "/")
            elif mode == "--nltk":
                replace(os.path.join(directory, wrong), "/", " ")

        if "Train" in file:
            if mode == "--stanford":

                fileName = "st_" + file.split(".")[0] + ".tagger"
                model = StanfordModel(fileName, os.path.join(directory, file))
                model.train()

                fileList = []
                directoriedFiles = []
                for f in files:
                    directoriedFiles.append(os.path.join(directory, f))

                    if "Test" in f:
                        fileList.append(f)

                results = testDirectory(model, directoriedFiles)
                i = 0
                for result in results:
                    fname = "st_temp_" + fileList[i]
                    f = open(fname, "wb+")
                    f.write(result)
                    f.close()
                    i += 1

            elif mode == "--nltk":
                tagList = prepareNLTK(os.path.join(directory, file))

                fileName = "nltk1.txt" if "Domain1" in directory else "nltk2.txt"

                modelHMM = HMM(directory=directory)
                trainedHMM = modelHMM.train(text=tagList)
                modelBrill = Brill(directory=directory, tagger=trainedHMM)
                modelBrill.train(text=tagList)

                testList = []
                fileList = []
                for testFile in files:
                    if "Test" in testFile:
                        testList.append(prepareNLTK(os.path.join(directory, testFile)))
                        fileList.append(testFile)

                hmmAcc = testDirectory(modelHMM, testList)
                brillAcc = testDirectory(modelBrill, testList)

                with open("results.txt", "a") as results:
                    i = 0
                    for acc in hmmAcc:
                        tagger = "Tagger: HMM   | "
                        testFile = fileList[i]
                        a = str(acc)
                        write = tagger + "Trained on: " + file + " | Tested on: " + testFile + " | Accuracy: " + a
                        results.write(write + "\n")
                        i += 1

                    i = 0
                    for acc in brillAcc:
                        tagger = "Tagger: Brill | "
                        testFile = fileList[i]
                        a = str(acc)
                        write = tagger + "Trained on: " + file + " | Tested on: " + testFile + " | Accuracy: " + a
                        results.write(write + "\n")
                        i += 1

            else:
                print("Incorrect usage: mode must be either --stanford or --nltk")



if __name__ == '__main__':
    #check command line arguments
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Usage: ./main.py <--stanford or --nltk> <directory to model>\n")