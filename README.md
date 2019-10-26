# cmput497_a3

Sources consulted:
https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
https://www.geeksforgeeks.org/nlp-brill-tagger/

To run the program, download the stanford POS tagger from https://nlp.stanford.edu/software/tagger.shtml#Download . Then, pull `stanford-postagger.jar`out of the extracted zip file. Ensure nltk is installed by running `pip3 install nltk`, then use `python3 ./code/main.py <--stanford or --nltk> <directory to be trained/tested>`

The results for the `nltk taggers` are found in `results.txt`, and the results from the Stanford tagger were manually gathered from the output, and can be found in `st_found_results.txt`.