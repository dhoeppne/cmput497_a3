# import external libraries
import sys, os, re
# import my files
from models import HMM, Brill, StanfordModel

def main(mode, directory):

    #try to create extraction directory, remove it if it exists
    if mode == "--stanford":
        fileName = "Stanford1.txt" if "Domain1" in directory else "Stanford2.txt"
        model = StanfordModel()
    elif mode == "--nltkHMM":
        model = HMM()
        fileName = "HMM1.txt" if "Domain1" in directory else "HMM2.txt"
    elif mode == "--nltkBrill":
        model = Brill()
        fileName = "Brill1.txt" if "Domain1" in directory else "Brill.txt"
    else:
        print("Incorrect usage: mode must be either --stanford or --nltkHMM or --nltkBrill")

    results = open(fileName, "w")

    # prep file
    with open(directory, "r") as file:
        print("open")
        for line in file:
            if re.search(r'\w+\s\w+', line):
                line = line.replace(" ", "/")

    model.train()

if __name__ == '__main__':
    #check command line arguments
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Usage: ./main.py <--stanford or --nltkHMM or --nltkBrill> <.txt file to model>\n")