import subprocess
from os import remove
class NltkModel(object):
    def __init__(self, file):
        self.file = file

class HMM(NltkModel):
    def __init__(self):
        print("HMM")

class Brill(NltkModel):
    def __init__(self):
        print("Brill")

class StanfordModel(object):
    def __init__(self, fileName, directory):
        self.fileName = fileName
        self.directory = directory

        try:
            remove("./st-temp.txt")
        except:
            print("File does not exist, continuing as normal")
        file = open("./st-temp.txt", "w")
        file.close()

    def train(self):
        bash_script = "java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop ./myPropsFile.prop -model "
        bash_script += self.fileName
        bash_script += " -trainFile "
        bash_script += self.directory

        process = subprocess.Popen(bash_script.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

