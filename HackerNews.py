# -------------------------------------------------------
# Assignment 2
# Written by Mushfiqur Anik
# For COMP 472 Section: JX – Summer 2020
# --------------------------------------------------------

# imports
import operator
import csv
import numpy as np
import re
import math
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class HackerNews:

    #================================HELPER FUNCTIONS===============================================
    # count number of occurences
    def count_occurrences(self, word, sentence):
        return sentence.count(word)

    # remove duplicate function
    def remove(self, duplicate):
        final_list = []
        for num in duplicate:
            if num not in final_list:
                final_list.append(num)
        return final_list

    # remove empty space from list
    def removeEmptySpace(self, myList):
        while ("" in myList):
            myList.remove("")
        return myList

    # These are the removed words for cleaning
    def setRemovedWords(self, removedWords):
        self.removedWords = removedWords
    def getRemovedWords(self):
        return self.removedWords

    # Set & Get Frequency dictionary word:frequencyOfWord
    def setFreqDictStory(self, freqDictStory):
        self.freqDictStory = freqDictStory
    def setFreqDictAsk(self, freqDictAsk):
        self.freqDictAsk = freqDictAsk
    def setFreqDictShow(self, freqDictShow):
        self.freqDictShow = freqDictShow
    def setFreqDictPoll(self, freqDictPoll):
        self.freqDictPoll = freqDictPoll

    def getFreqDictStory(self):
        return self.freqDictStory
    def getFreqDictAsk(self):
        return self.freqDictAsk
    def getFreqDictShow(self):
        return self.freqDictShow
    def getFreqDictPoll(self):
        return self.freqDictPoll

    # Set & Get Probability Dictionary word:probability of word in each class
    def setProbDictStory(self, probDictStory):
        self.probDictStory = probDictStory
    def setProbDictAsk(self, probDictAsk):
        self.probDictAsk = probDictAsk
    def setProbDictShow(self, probDictShow):
        self.probDictShow = probDictShow
    def setProbDictPoll(self, probDictPoll):
        self.probDictPoll = probDictPoll

    def getProbDictStory(self):
        return self.probDictStory
    def getProbDictAsk(self):
        return self.probDictAsk
    def getProbDictShow(self):
        return self.probDictShow
    def getProbDictPoll(self):
        return self.probDictPoll

    # Set and get smoothing
    def setSmoothing(self, smoothing):
        self.smoothing = smoothing
    def getSmoothing(self):
        return self.smoothing

    # Set and get totalWords in each class
    def setTotalWordsStory(self, totalWordsStory):
        self.totalWordsStory = totalWordsStory
    def setTotalWordsAsk(self, totalWordsAsk):
        self.totalWordsAsk = totalWordsAsk
    def setTotalWordsShow(self, totalWordsShow):
        self.totalWordsShow = totalWordsShow
    def setTotalWordsPoll(self, totalWordsPoll):
        self.totalWordsPoll = totalWordsPoll

    def getTotalWordsStory(self):
        return self.totalWordsStory
    def getTotalWordsAsk(self):
        return self.totalWordsAsk
    def getTotalWordsShow(self):
        return self.totalWordsShow
    def getTotalWordsPoll(self):
        return self.totalWordsPoll

    # vocabulary list for each class
    def setVocabStory(self, vocabStory):
        self.vocabStory = vocabStory
    def setVocabAsk(self, vocabAsk):
        self.vocabAsk = vocabAsk
    def setVocabShow(self, vocabShow):
        self.vocabShow = vocabShow
    def setVocabPoll(self, vocabPoll):
        self.vocabPoll = vocabPoll

    def getVocabStory(self):
        return self.vocabStory
    def getVocabAsk(self):
        return self.vocabAsk
    def getVocabShow(self):
        return self.vocabShow
    def getVocabPoll(self):
        return self.vocabPoll

    # Set and get how many times each class appeared in the dataset
    def setNumOfStory(self, numOfStory):
        self.numOfStory = numOfStory
    def setNumOfAsk(self, numOfAsk):
        self.numOfAsk = numOfAsk
    def setNumOfShow(self, numOfShow):
        self.numOfShow = numOfShow
    def setNumOfPoll(self, numOfPoll):
        self.numOfPoll = numOfPoll

    def getNumOfStory(self):
        return self.numOfStory
    def getNumOfAsk(self):
        return self.numOfAsk
    def getNumOfShow(self):
        return self.numOfShow
    def getNumOfPoll(self):
        return self.numOfPoll

    # Set and get Stopwords
    def setStopWords(self, file):
        stopWords = []
        newWord = ""

        with open(file) as f:
            lines =  [line.rstrip() for line in f]

        for i in range(0, len(lines)):
            newWord = "".join(re.split("[^a-zA-Z]*", lines[i]))
            stopWords.append(newWord)

        self.stopWords = stopWords

    def getStopWords(self):
        return self.stopWords

    # Remove all occorences of one list from another
    def removeAllOccurences(self, list1, list2):
        print("Inside removeAll occurences")
        for i in range(0,len(list2)):
            word = list2[i]
            list1 = list(filter((word).__ne__, list1))
        return list1

    # Experiment-2
    def removeLength(self, list):
        elementsToRemove = []
        word = " "

        for i in range(0, len(list)):
            word = list[i]
            if len(word) <= 2 or len(word) >= 9:
                elementsToRemove.append(word)

        print(elementsToRemove)
        list = self.removeAllOccurences(list, elementsToRemove)

        return list

    # Correct classifications
    def setCorrectClassification(self, correctClassification):
        self.correctClassification = correctClassification
    def getCorrectClassication(self):
        return self.correctClassification

    # Predicted classifications by my classifier
    def setPredictClassification(self, predictClassification):
        self.predictClassification = predictClassification

    def getPredictClassication(self):
        return self.predictClassification

    # Total frequency of word in all classes combined
    def setTotalFrequency(self, totalFrequency):
        self.totalFrequency = totalFrequency
    def getTotalFrequency(self):
        return self.totalFrequency

    # ================================MAIN FUNCTIONS===============================================
    # ---Extract Vocabulary list---
    def extractVocabulary(self, file, experiment):
        # Total of each class
        numOfStory = 0
        numOfAsk = 0
        numOfShow = 0
        numOfPoll = 0

        # Removed Words list
        removedWords = []

        # Vocab for each class
        vocabStory = []
        vocabAsk = []
        vocabShow = []
        vocabPoll = []

        counter = 0
        counter2 = 0

        vocabulary = []

        # Extracts the vocabulary after cleaning the data
        with open(file, 'r') as csv_file:

            csv_reader = csv.DictReader(csv_file)

            for line in csv_reader:
                if line['year'] == "2018":
                    counter2 += 1
                    x = line['Title'].split()

                    for i in range(len(x)):
                        x[i] = ''.join(i for i in x[i].lower())  # Lower case
                        removedWord = " ".join(re.split("[a-zA-Z]*", x[i]))
                        removedWord = removedWord.replace(" ", "")
                        removedWords.append(removedWord)
                        extractedWord = " ".join(re.split("[^a-zA-Z]*", x[i]))
                        extractedWord = extractedWord.replace(" ", "")
                        vocabulary.append(extractedWord)

                        if line['Post Type'] == "story":
                            numOfStory += 1
                            vocabStory.append(extractedWord)
                        elif line['Post Type'] == "ask_hn":
                            numOfAsk += 1
                            vocabAsk.append(extractedWord)
                        elif line['Post Type'] == "show_hn":
                            numOfShow += 1
                            vocabShow.append(extractedWord)
                        elif line['Post Type'] == "poll":
                            numOfPoll += 1
                            vocabPoll.append(extractedWord)

        # Setting number of each class
        self.setNumOfStory(numOfStory)
        self.setNumOfAsk(numOfAsk)
        self.setNumOfShow(numOfShow)
        self.setNumOfPoll(numOfPoll)

        # Removing empty spaces
        vocabulary = self.removeEmptySpace(vocabulary)
        removedWordsList = self.removeEmptySpace(removedWords)
        vocabStory = self.removeEmptySpace(vocabStory)
        vocabAsk = self.removeEmptySpace(vocabAsk)
        vocabShow = self.removeEmptySpace(vocabShow)
        vocabPoll = self.removeEmptySpace(vocabPoll)

        # Setting the vocabLists
        self.setVocabStory(vocabStory)
        self.setVocabAsk(vocabAsk)
        self.setVocabShow(vocabShow)
        self.setVocabPoll(vocabPoll)

        if experiment == "experiment3":
            return vocabulary

        # Remove duplicates from list
        vocabulary = self.remove(vocabulary)

        # Sort vocabulary alphabetically
        vocabulary = sorted(vocabulary)
        print(vocabulary)
        print(len(vocabulary))

        # Experiments

        # For experiment-1 we remove stopwords
        if experiment == "experiment1":
            print(experiment)
            stopWords = self.getStopWords()

            for i in range(0, len(stopWords)):
                removedWords.append(stopWords[i])
            vocabulary = self.removeAllOccurences(vocabulary,stopWords)

        # For experiment-2 we wordlength
        elif experiment=="experiment2":
            print(experiment)
            for i in range(0, len(vocabulary)):
                if len(vocabulary[i]) <= 2 or len(vocabulary[i]) >= 9:
                    removedWords.append(vocabulary[i])
            vocabulary = self.removeLength(vocabulary)

        self.setRemovedWords(removedWords)
        print(vocabulary)
        print(len(vocabulary))

        return vocabulary

    # ---Frequency & Probability---
    def frequencyCounter(self, file, vocabulary):

        # Vocabulary Length
        vocabLength = len(vocabulary)

        # Frequency Dictionary
        freqDictStory = {}
        freqDictAsk = {}
        freqDictShow = {}
        freqDictPoll = {}

        totalWordsStory = 0
        totalWordsAsk = 0
        totalWordsShow = 0
        totalWordsPoll = 0

        totalFrequency = {}

        # For each word how many times does it appear in each class
        for i in range(0, vocabLength):
            word = vocabulary[i]
            wordFreqStory = 0
            wordFreqAsk = 0
            wordFreqShow = 0
            wordFreqPoll = 0

            wordFreqStory = self.getVocabStory().count(word)
            wordFreqAsk = self.getVocabAsk().count(word)
            wordFreqShow = self.getVocabShow().count(word)
            wordFreqPoll = self.getVocabPoll().count(word)

            totalWordsStory += self.getVocabStory().count(word)
            totalWordsAsk += self.getVocabAsk().count(word)
            totalWordsShow += self.getVocabShow().count(word)
            totalWordsPoll += self.getVocabPoll().count(word)

            freqDictStory[word] = wordFreqStory
            freqDictAsk[word] = wordFreqAsk
            freqDictShow[word] = wordFreqShow
            freqDictPoll[word] = wordFreqPoll

            total = wordFreqStory + wordFreqAsk + wordFreqShow + wordFreqPoll
            totalFrequency[word] = total

        # Setting up the dictionaries
        self.setFreqDictStory(freqDictStory)
        self.setFreqDictAsk(freqDictAsk)
        self.setFreqDictShow(freqDictShow)
        self.setFreqDictPoll(freqDictPoll)

        self.setTotalWordsStory(totalWordsStory)
        self.setTotalWordsAsk(totalWordsAsk)
        self.setTotalWordsShow(totalWordsShow)
        self.setTotalWordsPoll(totalWordsPoll)

        self.setTotalFrequency(totalFrequency)

    # Calculating probabilities
    def calculateProbability(self, vocabulary):
        vocabLength = len(vocabulary)

        # Get frequencies
        freqDictStory = self.getFreqDictStory()
        freqDictAsk = self.getFreqDictAsk()
        freqDictShow = self.getFreqDictShow()
        freqDictPoll = self.getFreqDictPoll()

        probDictStory = {}
        probDictAsk = {}
        probDictShow = {}
        probDictPoll = {}

        # Total number of words in each class
        totalWordsStory = (self.getTotalWordsStory() + (self.getSmoothing() * vocabLength))
        totalWordsAsk = (self.getTotalWordsAsk() + (self.getSmoothing() * vocabLength))
        totalWordsShow = (self.getTotalWordsShow() + (self.getSmoothing() * vocabLength))
        totalWordsPoll = (self.getTotalWordsPoll() + (self.getSmoothing() * vocabLength))

        # Probability
        prob1 = 0
        prob2 = 0
        prob3 = 0
        prob4 = 0

        # Calculate the probabilities by adding smoothing
        for i in range(0, vocabLength):
            word = vocabulary[i]
            probDictStory[word] = (freqDictStory[word] + self.getSmoothing()) / totalWordsStory
            probDictAsk[word] = (freqDictAsk[word] + self.getSmoothing()) / totalWordsAsk
            probDictShow[word] = (freqDictShow[word] + self.getSmoothing()) / totalWordsShow
            probDictPoll[word] = (freqDictPoll[word] + self.getSmoothing()) / totalWordsPoll

            prob1 += probDictStory[word]
            prob2 += probDictAsk[word]
            prob3 += probDictShow[word]
            prob4 += probDictPoll[word]

        self.setProbDictStory(probDictStory)
        self.setProbDictAsk(probDictAsk)
        self.setProbDictShow(probDictShow)
        self.setProbDictPoll(probDictPoll)

        print("Smoothed total words for each class..")
        print(totalWordsStory)
        print(totalWordsAsk)
        print(totalWordsShow)
        print(totalWordsPoll)

    def model(self, file, modelFile, vocabularyFile, removeFile, vocabulary):
        vocabLength = len(vocabulary)

        # RemovedWords
        removedWords = self.getRemovedWords()

        # Frequencies
        freqDictStory = self.getFreqDictStory()
        freqDictAsk = self.getFreqDictAsk()
        freqDictShow = self.getFreqDictShow()
        freqDictPoll = self.getFreqDictPoll()

        # Probabilities
        probDictStory = self.getProbDictStory()
        probDictAsk = self.getProbDictAsk()
        probDictShow = self.getProbDictShow()
        probDictPoll = self.getProbDictPoll()

        # Writing the vocabulary.txt file
        with open(vocabularyFile, 'w') as f:
            for i in range(0, vocabLength):
                word = vocabulary[i]
                f.write(str(i) + "  ")
                f.write(vocabulary[i])
                f.write("\n")

        with open(removeFile, 'w') as f:
            for i in range(0, len(removedWords)):
                word = removedWords[i]
                f.write(str(i) + "  ")
                f.write(removedWords[i])
                f.write("\n")

        # Writing the model-2018.txt file
        with open(modelFile, 'w') as f:
            for i in range(0, vocabLength):
                word = vocabulary[i]
                f.write(str(i) + "  ")
                f.write(str(vocabulary[i]) + "  ")
                f.write(str(freqDictStory[word]) + "  ")
                f.write(str(probDictStory[word]) + "  ")
                f.write(str(freqDictAsk[word]) + "  ")
                f.write(str(probDictAsk[word]) + "  ")
                f.write(str(freqDictShow[word]) + "  ")
                f.write(str(probDictShow[word]) + "  ")
                f.write(str(freqDictPoll[word]) + "  ")
                f.write(str(probDictPoll[word]) + "  ")
                f.write("\n")

    # Calculate score of title in each class
    # Classify them if predicted classification == correct classification
    def calculateResults(self, file, resultFile, vocabulary, experiment):

        correctClassification = []
        predictClassification = []

        rightCounter = 0
        wrongCounter = 0
        maximum = 0
        classifiedClass = ""
        label = " "

        total = (self.numOfStory + self.numOfAsk + self.numOfShow + self.numOfPoll)
        probOfStory = self.getNumOfStory() / total
        probOfAsk = self.getNumOfAsk() / total
        probOfShow = self.getNumOfShow() / total
        probOfPoll = self.getNumOfPoll() / total

        # Probabilities
        probDictStory = self.getProbDictStory()
        probDictAsk = self.getProbDictAsk()
        probDictShow = self.getProbDictShow()
        probDictPoll = self.getProbDictPoll()

        # For each title in the dataset calculate it's score for each class
        with open(file, 'r') as csv_file:

            csv_reader = csv.DictReader(csv_file)
            title = []
            fullTitle = ""
            counter = -1

            for line in csv_reader:
                # Scores
                if probOfStory == 0:
                    scoreStory = float('-inf')
                else:
                    scoreStory = math.log(probOfStory, 10)

                if probOfAsk == 0:
                    scoreAsk = float('-inf')
                else:
                    scoreAsk = math.log(probOfAsk,10)

                if probOfShow == 0:
                    scoreShow = float('-inf')
                else:
                    scoreShow = math.log(probOfShow,10)

                if probOfPoll == 0:
                    scorePoll = float('-inf')
                else:
                    scorePoll = math.log(probOfPoll, 10)

                if line['year'] == "2019":
                    fullTitle = line['Title']
                    counter += 1
                    x = line['Title'].split()

                    for i in range(len(x)):
                        x[i] = ''.join(i for i in x[i].lower())  # Lower case
                        extractedWord = " ".join(re.split("[^a-zA-Z]*", x[i]))
                        extractedWord = extractedWord.replace(" ", "")
                        title.append(extractedWord)

                    for i in range(len(title)):
                        word = title[i]
                        if word in probDictStory:
                            scoreStory += math.log(probDictStory[word],10)
                        if word in probDictAsk:
                            scoreAsk += math.log(probDictAsk[word],10)
                        if word in probDictShow:
                            scoreShow += math.log(probDictShow[word],10)
                        if word in probDictStory:
                            scorePoll += math.log(probDictPoll[word],10)

                    maximum = max(scoreStory, scoreAsk, scoreShow, scorePoll)
                    #print(maximum)

                    if maximum == scoreStory:
                        classifiedClass = "story"
                    elif maximum == scoreAsk:
                        classifiedClass = "ask_hn"
                    elif maximum == scoreShow:
                        classifiedClass = "show_hn"
                    else:
                        classifiedClass = "poll"

                    if line['Post Type'] == classifiedClass:
                        label = "right"
                        rightCounter += 1
                    else:
                        label = "wrong"
                        wrongCounter += 1

                    # correctList
                    correctClassification.append(line['Post Type'])
                    # predictionList
                    predictClassification.append(classifiedClass)

                    if experiment != "experiment3":
                        with open(resultFile, 'a+') as r:
                            r.write(str(counter) + "  " + fullTitle + "  " + classifiedClass + "  " +
                                    "Story: " + str(scoreStory) + "  Ask: " + str(scoreAsk) + "  Show: " + str(
                                    scoreShow) + "  Poll: " + str(scorePoll) +
                                    "  " + line['Post Type'] + "  " + label + "\n")

                title = []

        print(rightCounter)
        print(wrongCounter)
        self.setCorrectClassification(correctClassification)
        self.setPredictClassification(predictClassification)

    def experiment3(self, file, vocabulary):

        experiment == "experiment3"
        resultFile = "something"

        vocabulary1 = []
        vocabulary2 = []

          # Get the vocubulary
        #--------- Frequency -----------
        print("=========================RUNNING EXPERIMENT-3.1==================================")
        fscore = []
        accuracy = []
        recall = []
        precision = []
        numOfWordsLeft = []

        correctClassification = []
        predictClassification = []

        # Frequency == 1
        for i in range(0,5):

            vocabulary1 = vocabulary.copy()

            print("Count number" + str(i))

            if i == 0:
                vocabulary1 = [i for i in vocabulary1 if vocabulary1.count(i) != 1]
            elif i == 1:
                vocabulary1 = [i for i in vocabulary1 if vocabulary1.count(i) >= 5]
            elif i == 2:
                vocabulary1 = [i for i in vocabulary1 if vocabulary1.count(i) >= 10]
            elif i == 3:
                vocabulary1 = [i for i in vocabulary1 if vocabulary1.count(i) >= 15]
            else:
                vocabulary1 = [i for i in vocabulary1 if vocabulary1.count(i) >= 20]

            print("Length of vocabulary after cleaning : " + str(len(vocabulary1)))
            vocabulary1 = self.remove(vocabulary1)

            print("Length of vocabulary after removing duplicates : " + str(len(vocabulary1)))
            print("Num of words left")

            numOfWordsLeft.append(len(vocabulary1))
            print(numOfWordsLeft)

            self.frequencyCounter(csvFile, vocabulary1)
            self.calculateProbability(vocabulary1)
            self.calculateResults(file, resultFile, vocabulary1, experiment)

            # calculate fscore, accurary, recall, precision and make a list
            correctClassification = self.getCorrectClassication()
            predictClassification = self.getPredictClassication()

            fscoreValue = f1_score(correctClassification, predictClassification, average='weighted')
            fscore.append(fscoreValue)

            recallValue = recall_score(correctClassification, predictClassification, average='weighted')
            recall.append(recallValue)

            precisionValue = precision_score(correctClassification, predictClassification, average='weighted')
            precision.append(precisionValue)

            accuracyValue = accuracy_score(correctClassification, predictClassification)
            accuracy.append(accuracyValue)

        print(fscore)
        print(recall)
        print(precision)
        print(accuracy)
        print(numOfWordsLeft)

        numOfWordsLeft0 = numOfWordsLeft.copy()

        print("")
        print("")
        print("")

        # ----------- Top frequency---------------
        print("=========================RUNNING EXPERIMENT-3.2==================================")
        fscore1 = []
        accuracy1 = []
        recall1 = []
        precision1 = []
        numOfWordsLeft = []

        correctClassification = []
        predictClassification = []

        totalFrequency = {}
        listOfWordsToRemove = []
        allWords = []

        vocabulary2 = vocabulary.copy()
        vocabulary2 = self.remove(vocabulary2)
        self.frequencyCounter(csvFile, vocabulary2)

        lengthOfVocab = len(vocabulary2)
        totalFrequency = self.getTotalFrequency()

        totalFrequency = dict(sorted(totalFrequency.items(), key=operator.itemgetter(1), reverse=True))

        print(totalFrequency)

        allWords = list(totalFrequency)

        # Frequency == 1
        for i in range(0, 5):

            listOfWordsToRemove = []
            vocabulary2 = vocabulary.copy()

            print("Experiment number" + str(i))
            print("Length of vocabulary: " + str(len(vocabulary2)))

            if i == 0:
                # Top 5% will be removed
                removeUntil = math.floor(0.05*lengthOfVocab)
                print("Remove until : " + str(removeUntil))
                for j in range(0, removeUntil):
                    listOfWordsToRemove.append(allWords[j])

                print(listOfWordsToRemove)
                print(len(listOfWordsToRemove))
                vocabulary2 = self.removeAllOccurences(vocabulary2, listOfWordsToRemove)

            elif i == 1:
                # Top 10% will be removed
                removeUntil = math.floor(0.10 * lengthOfVocab)
                print("Remove until : " + str(removeUntil))
                for j in range(0, removeUntil):
                    listOfWordsToRemove.append(allWords[j])

                vocabulary2 = self.removeAllOccurences(vocabulary2, listOfWordsToRemove)

            elif i == 2:
                # Top 15% will be removed
                removeUntil = math.floor(0.15 * lengthOfVocab)
                print("Remove until : " + str(removeUntil))
                for j in range(0, removeUntil):
                    listOfWordsToRemove.append(allWords[j])

                vocabulary2 = self.removeAllOccurences(vocabulary2, listOfWordsToRemove)

            elif i == 3:
                # Top 20% will be removed
                removeUntil = math.floor(0.20 * lengthOfVocab)
                print("Remove until : " + str(removeUntil))
                for j in range(0, removeUntil):
                    listOfWordsToRemove.append(allWords[j])

                vocabulary2 = self.removeAllOccurences(vocabulary2, listOfWordsToRemove)

            else:
                # Top 25% will be removed
                removeUntil = math.floor(0.25 * lengthOfVocab)
                print("Remove until : " + str(removeUntil))
                for j in range(0, removeUntil):
                    listOfWordsToRemove.append(allWords[j])

                vocabulary2 = self.removeAllOccurences(vocabulary2, listOfWordsToRemove)

            print("Length of vocabulary after cleaning : " + str(len(vocabulary2)))
            vocabulary2 = self.remove(vocabulary2)

            print("Length of vocabulary after removing duplicates : " + str(len(vocabulary2)))
            print("Also num of words left")

            numOfWordsLeft.append(len(vocabulary2))
            print(numOfWordsLeft)

            self.frequencyCounter(csvFile, vocabulary2)
            self.calculateProbability(vocabulary2)
            self.calculateResults(file, resultFile, vocabulary2, experiment)

            # calculate fscore, accurary, recall, precision and make a list
            correctClassification = self.getCorrectClassication()
            predictClassification = self.getPredictClassication()

            fscoreValue = f1_score(correctClassification, predictClassification, average='weighted')
            fscore1.append(fscoreValue)

            recallValue = recall_score(correctClassification, predictClassification, average='weighted')
            recall1.append(recallValue)

            precisionValue = precision_score(correctClassification, predictClassification, average='weighted')
            precision1.append(precisionValue)

            accuracyValue = accuracy_score(correctClassification, predictClassification)
            accuracy1.append(accuracyValue)

        # Plot graph here
        print(fscore1)
        print(recall1)
        print(precision1)
        print(accuracy1)
        print(numOfWordsLeft)

        # Plot the graph now using matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.scatter(numOfWordsLeft0, fscore, color='r')
        ax1.scatter(numOfWordsLeft0, recall, color='b')
        ax1.scatter(numOfWordsLeft0, precision, color='y')
        ax1.scatter(numOfWordsLeft0, accuracy, color='c')
        ax1.set_xlabel('Number of words left')
        ax1.set_ylabel('Performance')
        ax1.set_title('words = & <= frequency removed')

        ax2.scatter(numOfWordsLeft, fscore1, color='r')
        ax2.scatter(numOfWordsLeft, recall1, color='b')
        ax2.scatter(numOfWordsLeft, precision1, color='y')
        ax2.scatter(numOfWordsLeft, accuracy1, color='c')
        ax2.set_xlabel('Number of words left')
        ax2.set_ylabel('Performance')
        ax2.set_title('%Top frequencies removed')
        fig.tight_layout(pad=3.0)

        legend_x = 1
        legend_y = 0.5

        plt.legend(["fscore", "recall", "precision", "accuracy"], loc='center left',
                   bbox_to_anchor=(legend_x, legend_y))
        plt.show()


#==================================== START OF PROGRAM =================================================================
csvFile = "/Users/mushfiquranik/Documents/TestingA2/hns_2018_2019.csv"
model2018 = "/Users/mushfiquranik/Documents/TestingA2/model-2018.txt"
stopWords = "/Users/mushfiquranik/Documents/TestingA2/stopwords.txt"
vocabularyFile = "/Users/mushfiquranik/Documents/TestingA2/vocabulary.txt"
removeFile = "/Users/mushfiquranik/Documents/TestingA2/remove.txt"
resultFile = "/Users/mushfiquranik/Documents/TestingA2/baseline-result.txt"
vocabulary = []

h1 = HackerNews()
h1.setSmoothing(0.5)
h1.setStopWords(stopWords)
experiment = "experiment"

#--------------------------------- BASELINE EXPERIMENT ----------------------------------------------
print("========================== BASELINE EXPERIMENT ==============================================")

#-------Testing vocabulary--------
print("-------Testing vocabulary--------")
vocabulary = h1.extractVocabulary(csvFile, experiment)

#------Calculate frequency---------
print("-------Testing Frequency--------")
h1.frequencyCounter(csvFile,vocabulary)

#----Calculate probability--------
print("-------Testing probabilities --------")
h1.calculateProbability(vocabulary)

# -----Testing model--------------
print("-------Testing model --------")
h1.model(csvFile, model2018, vocabularyFile, removeFile, vocabulary)

# ----Testing results-------
print("------_Testing results----------------")
h1.calculateResults(csvFile, resultFile, vocabulary, experiment)
print("-------------------------------------------------------------------------------------------")

#--------------------------------- EXPERIMENT-1 Stopword Word Filtering ------------------------------------------
print("========================== EXPERIMENT-1 Stopword Word Filtering ===========================================")
vocabularyFile = "/Users/mushfiquranik/Documents/TestingA2/stopword-vocabulary.txt"
removeFile = "/Users/mushfiquranik/Documents/TestingA2/stopword-remove.txt"
resultFile = "/Users/mushfiquranik/Documents/TestingA2/stopword-result.txt"
model2018 = "/Users/mushfiquranik/Documents/TestingA2/stopword-model.txt"

#-------Testing vocabulary--------
experiment = "experiment1"
print("-------Testing vocabulary experiment-1--------")
vocabulary = h1.extractVocabulary(csvFile, experiment)

#------Calculate frequency---------
print("-------Testing Frequency--------")
h1.frequencyCounter(csvFile,vocabulary)

#----Calculate probability--------
print("-------Testing probabilities --------")
h1.calculateProbability(vocabulary)

# -----Testing model--------------
print("-------Testing model --------")
h1.model(csvFile, model2018, vocabularyFile, removeFile, vocabulary)

# ----Testing results-------
print("------_Testing results----------------")
h1.calculateResults(csvFile, resultFile, vocabulary, experiment)
print("-------------------------------------------------------------------------------------------")


#--------------------------------- EXPERIMENT-2 Word Length Word Filtering ------------------------------------------
print("========================== EXPERIMENT-2 Word Length Word Filtering ===========================================")
vocabularyFile = "/Users/mushfiquranik/Documents/TestingA2/wordlength-vocabulary.txt"
removeFile = "/Users/mushfiquranik/Documents/TestingA2/wordlength-remove.txt"
resultFile = "/Users/mushfiquranik/Documents/TestingA2/wordlength-result.txt"
model2018 = "/Users/mushfiquranik/Documents/TestingA2/wordlength-model.txt"

#-------Testing vocabulary--------
experiment = "experiment2"
print("-------Testing vocabulary experiment-2--------")
vocabulary = h1.extractVocabulary(csvFile, experiment)

#------Calculate frequency---------
print("-------Testing Frequency--------")
h1.frequencyCounter(csvFile,vocabulary)

#----Calculate probability--------
print("-------Testing probabilities --------")
h1.calculateProbability(vocabulary)

# -----Testing model--------------
print("-------Testing model --------")
h1.model(csvFile, model2018, vocabularyFile, removeFile, vocabulary)

# ----Testing results-------
print("------_Testing results----------------")
h1.calculateResults(csvFile, resultFile, vocabulary, experiment)
print("-------------------------------------------------------------------------------------------")

#--------------------------------- EXPERIMENT-3 Infrequent Word Filtering ------------------------------------------
print("========================== EXPERIMENT-3 Infrequent Word Filtering ===========================================")
experiment = "experiment3"
vocabulary = h1.extractVocabulary(csvFile,experiment)
h1.experiment3(csvFile, vocabulary)