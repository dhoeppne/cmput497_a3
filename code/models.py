import subprocess
from os import remove
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag import brill
class NltkModel(object):
    def __init__(self, **kwargs):
        self.text = kwargs["text"]
        self.directory = kwargs["directory"]
        self.testText = kwargs["testText"]

class HMM(NltkModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hmm = HiddenMarkovModelTagger

    def train(self):
        self.hmmTrained = self.hmm.train(self.text)
        return self.hmmTrained

    def test(self):
        acc = self.hmmTrained.evaluate(self.testText)
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
    def train(self):
        self.trainedBrill = self.brillTrainer.train(self.text)

    def test(self):
        acc = self.trainedBrill.evaluate(self.testText)
        return acc

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

    def test(self, testFile):
        bash_script = "java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop ./myPropsFile.prop -model "
        bash_script += self.fileName + " -testFile " + testFile

        process = subprocess.Popen(bash_script.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
