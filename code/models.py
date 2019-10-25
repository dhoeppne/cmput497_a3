import subprocess
from os import remove
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag import brill
class NltkModel(object):
    def __init__(self, **kwargs):
        self.directory = kwargs["directory"]

class HMM(NltkModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hmm = HiddenMarkovModelTagger

    def train(self, text):
        self.hmmTrained = self.hmm.train(text)
        return self.hmmTrained

    def test(self, testText):
        acc = self.hmmTrained.evaluate(testText)
        return acc

class Brill(NltkModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tagger = kwargs["tagger"]

        templates = [
            brill.Template(brill.Pos([-1])),
            brill.Template(brill.Pos([1])),
            brill.Template(brill.Pos([-2])),
            brill.Template(brill.Pos([2])),
            brill.Template(brill.Pos([-2, -1])),
            brill.Template(brill.Pos([1, 2])),
            brill.Template(brill.Pos([-3, -2, -1])),
            brill.Template(brill.Pos([1, 2, 3])),
            brill.Template(brill.Pos([-1]), brill.Pos([1])),
            brill.Template(brill.Word([-1])),
            brill.Template(brill.Word([1])),
            brill.Template(brill.Word([-2])),
            brill.Template(brill.Word([2])),
            brill.Template(brill.Word([-2, -1])),
            brill.Template(brill.Word([1, 2])),
            brill.Template(brill.Word([-3, -2, -1])),
            brill.Template(brill.Word([1, 2, 3])),
            brill.Template(brill.Word([-1]), brill.Word([1])),
            ]

        self.brillTrainer = BrillTaggerTrainer(self.tagger, templates, deterministic=True)
    def train(self, text):
        self.trainedBrill = self.brillTrainer.train(text)

    def test(self, testText):
        acc = self.trainedBrill.evaluate(testText)
        return acc

class StanfordModel(object):
    def __init__(self, fileName, directory):
        self.fileName = fileName
        self.directory = directory

        try:
            remove("./st_results.txt")
        except:
            print("File does not exist, continuing as normal")
        file = open("./st_results.txt", "w")
        file.close()

    def train(self):
        bash_script = "java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop ./myPropsFile.prop -model "
        bash_script += self.fileName
        bash_script += " -trainFile "
        bash_script += self.directory

        process = subprocess.Popen(bash_script.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    def test(self, testFile):
        if "Test" in testFile:
            bash_script = "java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop ./myPropsFile.prop -model "
            bash_script += self.fileName + " -testFile " + testFile + " > ./st_results.txt"

            process = subprocess.Popen(bash_script.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            return 0
