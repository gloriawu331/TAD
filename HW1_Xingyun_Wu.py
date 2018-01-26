#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:40:09 2018

@author: Xingyun Wu

Homework 1 for Machine Learning
"""


from bs4 import BeautifulSoup
from urllib.request import urlopen
import re #, os
from nltk import word_tokenize
from nltk import trigrams
from nltk.stem import PorterStemmer
pt = PorterStemmer()
import csv


##### PROBLEM 1 #####

# Read in the debate
url  = urlopen('file:///Users/hsswyx/Desktop/machine_learning/TAD/Debate1.html').read()
soup = BeautifulSoup(url, "lxml")
pTags = soup.find_all('p')

def get_speaker_statement(whole_str):
    '''
    This auxiliary function is to help seperate a whole speaker_statement 
    string into speaker and statement.
    
    Input:
        whole_str: a string, the string that consists speaker and statement
    
    Outputs:
        rv: a list, containing speaker and speech, returned if the whole_str is
            in a "speaker: speech" format
        whole_str: the original string, returned if it is not in a
            "speaker: speech" format
    '''
    rv = []
    item_groups = re.search('([A-Z]+)\:\s(.*)', whole_str)
    if item_groups != None:
        speaker = item_groups.group(1)
        speech = item_groups.group(2)
        rv = [speaker, speech]
        return rv
    else:
        return whole_str

def merge_info(lst, flag, whole_str):
    '''
    This function helps to structure the debate.
    
    Inputs:
        lst: a list of lists, containing speakers and corresponding statements
        flag: an integer, 0 if no need to check speakers, 1 if need to check
            speakers
        whole_str: a string, a whole paragraph of the document
    '''
    info = get_speaker_statement(whole_str)
    if type(info) == list:
        speaker = info[0]
        statement = info[1]
        if flag == 1:
            if speaker == lst[-1][0]:
                lst[-1][1] = lst[-1][1] + ' ' + statement
            else:
                lst.append(info)
            flag = 0
        else:
            lst.append(info)
    elif type(info) == str:
        lst[-1][1] = lst[-1][1] + ' ' + info

# Build the nested lists
content = []
flag = 0
for item in pTags[6:-3]:
    item_text = item.text
    item_groups = re.search('(.*)(\([A-Z]+\))(.*)', item_text)
    if item_groups == None:
        merge_info(content, flag, item_text)
    else:
        if item_groups.group(2) == '(CROSSTALK)':
            flag = 1
            if item_groups.group(1) != '':
                merge_info(content, flag, item_groups.group(1))
            if item_groups.group(3) != '':
                merge_info(content, flag, item_groups.group(3))
        else:
            pass


### Task i) and ii) ###
for i in range(len(content)):
    content[i].append(i)


### Preparation for Task iii) and Task iv) ###
stop_word_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a11-smart-stop-list/english.stop'
stop_words = urlopen(stop_word_url).read().decode("utf-8").split('\n')
for i in range(len(stop_words)):
    stop_words[i] = re.sub(r'[^\w\s]', '', stop_words[i])

# Create four dictionaries to count unigrams and trigrams at a time:
#    1. unigrams_all: to count unigrams in the whole debate
#    2. unigrams_statement: to count unigrams for every one of the statement
#    3. trigrams_all: to count trigrams in the whole debate
#    4. trigrams_statement: to count trigrams for every one of the statement

# Create empty dictionaries
unigrams_all = {}
unigrams_statement = {}
trigrams_all = {}
trigrams_statement = {}

# Go through all the pTags
for item in content:
    temp = []
    unigrams_statement[item[2]] = [item[0], {}]
    trigrams_statement[item[2]] = [item[0], {}]
    s = re.sub(r'[^\w\s]', '', item[1]).lower()
    s = word_tokenize(s)
    for_tri = []
    # Go through words in the statement and find the unigrams
    for word in s:
        if word not in stop_words:
            word = pt.stem(word)
            for_tri.append(word)
            unigrams_all[word] = unigrams_all.get(word, 0) + 1
            unigrams_statement[item[2]][1][word] = unigrams_statement[item[2]][1].get(word, 0) + 1
    # Find the trigrams
    trigrams_pre = trigrams(for_tri)
    trigrams_content = []
    for thing in trigrams_pre:
        trigrams_content.append(thing)
    for thing in trigrams_content:
        trigrams_all[thing] = trigrams_all.get(thing, 0) + 1
        trigrams_statement[item[2]][1][thing] = trigrams_statement[item[2]][1].get(thing, 0) + 1


### Task iii) ###

# Find the 1000 most used unigrams
i = 0
unigrams_1000 = []
for item in sorted(unigrams_all, key=unigrams_all.get, reverse=True):
    if i < 1000:
#        print(item, unigrams_all[item])
        unigrams_1000.append(item)
        i += 1
    
# Find the 500 most used trigrams
j = 0
trigrams_500 = []
for item in sorted(trigrams_all, key=trigrams_all.get, reverse=True):
    if j < 500:
        trigrams_500.append(item)
        j += 1


### Task iv) ###
m = []
for i in range(len(content)):
    temp = []
    temp.append(content[i][2])
    temp.append(content[i][0])
    for j in range(len(unigrams_1000)):
        if unigrams_1000[j] in unigrams_statement[i][1].keys():
            temp.append(unigrams_statement[i][1][unigrams_1000[j]])
        else:
            temp.append(0)
    for k in range(len(trigrams_500)):
        if trigrams_500[k] in trigrams_statement[i][1].keys():
            temp.append(trigrams_statement[i][1][trigrams_500[k]])
        else:
            temp.append(0)
    m.append(temp)


### Task v) ###

# Create a line for variable names
var_names = ['statement_num', 'speaker']
var_names += unigrams_1000
var_names += trigrams_500

# Add the variable name list in the matrix
m.insert(0, var_names)

# Write the matrix into csv file
with open('matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(m)

