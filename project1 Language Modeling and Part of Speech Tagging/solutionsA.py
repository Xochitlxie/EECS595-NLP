import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_tuples = []
    bigram_tuples = []
    trigram_tuples = []
    #construct the n-gram list
    for sentence in training_corpus:
        unigram = str.split(sentence)
        unigram.append(STOP_SYMBOL)
        unigram_tuples += unigram
        unigram.insert(0,START_SYMBOL)
        bigram = list(nltk.bigrams(unigram))
        bigram_tuples += bigram
        unigram.insert(0,START_SYMBOL)
        trigram = list(nltk.trigrams(unigram))
        trigram_tuples += trigram
    #count the size of each n-gram list
    count_sentence = len(training_corpus)
    nums_unigram = len(unigram_tuples)
    #probability of unigram dictonary
    count_unigram = {}
    unigram_p = {}
    for uni in unigram_tuples:
        count_unigram[(uni,)] = count_unigram.get((uni,),0) + 1.0
    for i in count_unigram:
        prob_unigram = math.log(count_unigram[i]*1.0/nums_unigram,2)
        unigram_p[i] = prob_unigram

    #probability of bigram dictonary
    count_bigram = {}
    bigram_p={}
    for bigram in bigram_tuples:
        count_bigram[bigram] = count_bigram.get(bigram,0) + 1.0
    for j in count_bigram:
        if j[0] == START_SYMBOL:
            bigram_p[j] = math.log(count_bigram[j]*1.0/count_sentence,2)
        else:
            bigram_p[j] = math.log(count_bigram[j]*1.0/count_unigram[(j[0],)],2)

    #probability of bigram dictonary
    count_trigram = {}
    trigram_p = {}
    for trigram in trigram_tuples:
        count_trigram[trigram] = count_trigram.get(trigram,0) + 1.0
    for k in count_trigram:
        if k[0]==START_SYMBOL and k[1]==START_SYMBOL:
            trigram_p[k] = math.log(count_trigram[k]*1.0/count_sentence,2)
        else:
            trigram_p[k] = math.log(count_trigram[k]*1.0/count_bigram[k[0:2]],2)

    return unigram_p, bigram_p, trigram_p
# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')


    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []
    #Unigram
    if n==1:
        for sentence in corpus:
            unigram = str.split(sentence)
            unigram.append(STOP_SYMBOL)
            prob_unisentence = 0.0
            for uni in unigram:
                if (uni,) not in ngram_p:
                    scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
                else:
                    prob_unisentence += ngram_p[(uni,)]
            if len(scores)==0 or scores[-1] != -1000:
                scores.append(prob_unisentence)

    #Bigram
    elif n==2:
        for sentence in corpus:
            unigram = str.split(sentence)
            unigram.append(STOP_SYMBOL)
            unigram.insert(0,START_SYMBOL)
            bigram = list(nltk.bigrams(unigram))
            prob_bisentence= 0.0
            for bi in bigram:
                if bi not in ngram_p:
                    scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
                else:
                    prob_bisentence += ngram_p[bi]
            if len(scores)==0 or scores[-1] != -1000:
                scores.append(prob_bisentence)


    #Trigram
    else:
        for sentence in corpus:
            unigram = str.split(sentence)
            unigram.append(STOP_SYMBOL)
            unigram.insert(0,START_SYMBOL)
            unigram.insert(0,START_SYMBOL)
            trigram = list(nltk.trigrams(unigram))
            prob_trisentence= 0.0
            for tri in trigram:
                if tri not in ngram_p:
                    scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                    break
                else:
                    prob_trisentence += ngram_p[tri]
            if len(scores)==0 or scores[-1] != -1000:
                scores.append(prob_trisentence)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for sentence in corpus:
        tokens = str.split(sentence)
        unigram = tokens + [STOP_SYMBOL]
        bigram_tokens = [START_SYMBOL] + unigram
        bigram = list(nltk.bigrams(bigram_tokens))
        trigram_tokens = [START_SYMBOL] + bigram_tokens
        trigram = list(nltk.trigrams(trigram_tokens))
        lin_score = 0
        for index in range(0,len(unigram)):
            if (unigram[index],) in unigrams and bigram[index] in bigrams and trigram[index] in trigrams:
                pro_score = (1.0/3)*2**(unigrams[(unigram[index],)]) + (1.0/3)*2**(bigrams[bigram[index]]) + (1.0/3)*2**(trigrams[trigram[index]])
                lin_score += math.log(pro_score,2)
            else:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                break
        if len(scores)==0 or scores[-1] != -1000:
            scores.append(lin_score)

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
