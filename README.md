# HackerNews
This repository contains the files for HackerNews project done for the Artificial intelligence course comp-472 offered by Concordia University(Montreal, Canada)

## Description 
This project contains a dataset of Hacker News fetched from kaggle. Hacker News is a popular technology site, where user-submitted stories (known as "posts") are
voted and commented upon. The site is extremely popular in technology and start-up circles. The top posts can attract hundreds of thousands of visitors.

In this program we build a probabilities model from the the training set and build a probabilistic model from the training set. The code will parse the file in the training set and build a vocabulary with all the words it contains in Title which is Created At 2018. Then for each word, compute their frequencies and the probabilities of each Post Type class (story, ask_hn, show_hn and poll). Extract the data from Created At 2019 as the testing dataset. 

#1.3 Task 3: Experiments with the classifier
Tasks 1 & 2 above will constitute your experiment 1, or baseline experiment, and you will perform
variations over this baseline to see if they improve the performance of your classifier.

#1.3.1 Experiment 1: Stop-word Filtering
Download the list of stop words available on Moodle. Use the baseline experiment and redo tasks
1 and 2 but this time remove the stop words from your vocabulary. Generate the new model and
result files that you will call stopword-model.txt and stopword-result.txt.

#1.3.2 Experiment 2: Word Length Filtering
Use the baseline experiment and redo tasks 1 and 2 but this time remove all words with length ≤
2 and all words with length ≥ 9. Generate the new model and result files that you will call
wordlength-model.txt and wordlength-result.txt.

#1.3.3 Experiment 3: Infrequent Word Filtering
Use the baseline experiment, and gradually remove from the vocabulary words with frequency=
1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20. Then gradually remove the
top 5% most frequent words, the 10% most frequent words, 15%, 20% and 25% most frequent
words. Plot both performance of the



## File List

## Built with
Python
PyCharm

## Author(s)

* [**Mushfiqur Anik**](https://github.com/mushfiqur-anik)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details





 
