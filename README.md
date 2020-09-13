# Naive-Bayes-Classifier
This repository contains the files for NaiveBayesClassifier project done for the Artificial intelligence course comp-472 offered by Concordia University(Montreal, Canada)

## Description 
The purpose of this project was to learn and apply the concepts of Natural Language Processing(NPL) & Machine Learning(ML). In this project a dataset (Dataset of Kaggle taken from kaggle) was provided and it consisted of posts from users from year 2018-19. 
Each post includes the following columns: 
- (Object ID | Title | Post Type | Author | Created At | URL | Points | Number of Comments | year)

##### Task:1 Extract the data and build the model
In this task a probabilistic model was built by tokenizing the titles Created At 2018 (Used as the training data). The model consisted of each tokenized word followed by the frequencies & smoothed conditional probabilities of eadch Post Type (story, ask_hn, show_hn, & poll respectively). The model file model-2018.txt looks like:
- 1 block 3 0.003 40 0.4 10 0.014 4 0.04
- 2 query 40 0.4 50 0.03 20 0.00014 15 0.4

##### Task:2 Use Machine Learning Classifier (Naive-Bayes-Classifier) to test dataset
In this task test our Naive-Bayes-Classifier from the training dataset training dataset to classify posts taken from 2019 into their likely class. The testing results were saved into the baseline-result.txt file. The file consists of each Title followed by the classification given by the classifier, score of each class (story, ask_hn, show_hn, & poll respectively), the correct classification, the label right or wrong. For example: 
- 1 Y Combinator story 0.004 0.001 0.0002. 0.002 story right
- 2 A Student's Guide poll 0.002 0.03 0.007 0.12 story wrong

##### Task:3 Experiments with the classifier
Different variations were performed over the baseline and the above tasks(Task-1 & Task-2) were implemented with the following constraints:
- 3.1 Stop words filtering - All the words in the stopword.txt file were removed and task-1 & 2 were performed again. The new model & results were saved in                                      stopword-model.txt and stopword-result.txt files respectively.
- 3.2 Word Length filtering - All the words with length ≤ 2 & length ≥ 9 were removed task-1 & 2 were performed again. The new model & results were saved in                                      wordlength-model.txt and wordlength-result.txt files respectively. 
- 3.3 Infrequent Word filtering - 




## File List
- HackerNews.py
- hns_2018_2019.csv
- stopwords.txt

## Built with
* [**Python**](https://en.wikipedia.org/wiki/Python_(programming_language)) - The Programming language used
* [**PyCharm**](https://en.wikipedia.org/wiki/PyCharm) - The IDE used

## Author(s)

* [**Mushfiqur Anik**](https://github.com/mushfiqur-anik)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details





 
