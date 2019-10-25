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

def main(mode, directory):
    # prep file
    replace(directory, " ", "/")

    #try to create extraction directory, remove it if it exists
    if mode == "--stanford":
        fileName = "Stanford1.tagger" if "Domain1" in directory else "Stanford2.tagger"
        model = StanfordModel(fileName, directory)
    elif mode == "--nltkHMM":
        model = HMM(fileName, directory)
        fileName = "HMM1.txt" if "Domain1" in directory else "HMM2.txt"
    elif mode == "--nltkBrill":
        model = Brill(fileName, directory)
        fileName = "Brill1.txt" if "Domain1" in directory else "Brill.txt"
    else:
        print("Incorrect usage: mode must be either --stanford or --nltkHMM or --nltkBrill")

    results = open(fileName, "w+")

    model.train()

if __name__ == '__main__':
    #check command line arguments
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Usage: ./main.py <--stanford or --nltkHMM or --nltkBrill> <.txt file to model>\n")