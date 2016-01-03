import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        brown_wordslist = [START_SYMBOL]
        brown_tagslist = [START_SYMBOL]
        tokens = str.split(sentence)
        for words in tokens:
            for i in range(len(words)-1,-1,-1):
                if words[i]=="/":
                    brown_wordslist.append(words[:i])
                    brown_tagslist.append(words[i+1:])
                    break
        brown_tagslist.append(STOP_SYMBOL)
        brown_wordslist.append(STOP_SYMBOL)
        brown_tags.append(brown_tagslist)
        brown_words.append(brown_wordslist)
    return brown_words, brown_tags


# TODO:IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    uni_tuples = []
    bi_tuples = []
    tri_tuples = []
    count_sentence = len(brown_tags)
    for sentence in brown_tags:
        bigram = list(nltk.bigrams(sentence))
        bi_tuples += bigram
        tri_tokens = [START_SYMBOL] + sentence
        trigram = list(nltk.trigrams(tri_tokens))
        tri_tuples += trigram
        unigram = sentence
        uni_tuples += unigram

    #probability of unigram tags
    count_unigram = {}
    unigram_p = {}
    nums_unigram = len(uni_tuples)-count_sentence
    for uni in uni_tuples:
        count_unigram[(uni,)] = count_unigram.get((uni,),0) + 1.0
    for i in count_unigram:
        if i != START_SYMBOL:
            prob_unigram = math.log(count_unigram[i]*1.0/nums_unigram,2)
            unigram_p[i] = prob_unigram
    #pribability of bigram tags
    count_bigram = {}
    bigram_p={}
    for bigram in bi_tuples:
        count_bigram[bigram] = count_bigram.get(bigram,0) + 1.0
    for j in count_bigram:
        if j[0] == START_SYMBOL:
            bigram_p[j] = math.log(count_bigram[j]*1.0/count_sentence,2)
        else:
            bigram_p[j] = math.log(count_bigram[j]*1.0/count_unigram[(j[0],)],2)


    #probability of trigram tags
    q_values = {}
    count_tritags = {}
    for tri in tri_tuples:
        count_tritags[tri] = count_tritags.get(tri,0) + 1.0
    for k in count_tritags:
        if k[0]==START_SYMBOL and k[1]==START_SYMBOL:
            q_values[k] = math.log(count_tritags[k]*1.0/count_sentence,2)
        else:
            q_values[k] = math.log(count_tritags[k]*1.0/count_bigram[k[0:2]],2)

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    count_uni={}
    known_words = set([])
    for sentence in brown_words:
        sentence = [START_SYMBOL] + sentence
        for word in sentence:
            count_uni[word] = count_uni.get(word,0) + 1.0
    for key in count_uni:
        if count_uni[key]>RARE_WORD_MAX_FREQ:
            known_words.add(key)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sentence in brown_words:
        sentences = [START_SYMBOL]
        for words in sentence:
            if words in known_words:
                sentences.append(words)
            else:
                sentences.append(RARE_SYMBOL)
        brown_words_rare.append(sentences)
    return brown_words_rare


# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_value = {}
    taglist = set([])
    tag_count = {}
    for b in brown_words_rare:
        del b[1]
    for a in brown_tags:
        a = [START_SYMBOL] + a
    for i in range(0,len(brown_words_rare)):
        for j in range(0,len(brown_words_rare[i])):
            word = brown_words_rare[i][j]
            tag = brown_tags[i][j]
            taglist.add(tag)
            e_value[(word, tag)] = e_value.get((word, tag), 0) + 1
            tag_count[tag] = tag_count.get(tag, 0) + 1
    for (pair, val) in e_value.items():
        e_value[pair] = math.log(1.0 * val / tag_count[pair[1]], 2)

    return e_value, taglist


# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!


def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    tL = list(taglist)
    tagSize = len(tL)

    #tran = [[LOG_PROG_OF_ZERO] * tagSize] * tagSize


    for sentense in brown_dev_words:
        #initialization
        sentense.append(STOP_SYMBOL)
        prob = [[]]
        back = []

        for i in range(tagSize * tagSize):
           tag1 = i / tagSize
           tag2 = i % tagSize
           if(tL[tag1] == START_SYMBOL and tL[tag2] == START_SYMBOL):
                prob[0].append(0)
           else:
                prob[0].append(LOG_PROB_OF_ZERO)

        for t in range(len(sentense)):
            backTrack = []
            newProb = []
            word = sentense[t]
            if word not in known_words:
                word = RARE_SYMBOL
            for state in range(tagSize * tagSize):
                t2 = state / tagSize
                t3 = state % tagSize
                if(e_values.get((word, tL[t3]), LOG_PROB_OF_ZERO) == LOG_PROB_OF_ZERO):
                    newProb.append(LOG_PROB_OF_ZERO)
                    backTrack.append(-1)
                    continue
                maxPairProb = LOG_PROB_OF_ZERO
                maxT1 = 0

                for t1 in range(tagSize):
                    priorState = t1 * tagSize + t2
                    pairProb = prob[t][priorState] + q_values.get((tL[t1], tL[t2], tL[t3]), LOG_PROB_OF_ZERO) + e_values.get((word, tL[t3]), LOG_PROB_OF_ZERO)
                    if pairProb > maxPairProb:
                        maxPairProb = pairProb
                        maxT1 = t1

                backTrack.append(maxT1)
                newProb.append(maxPairProb)
            prob.append(newProb)
            back.append(backTrack)

        maxEnd = 0
        maxProb = LOG_PROB_OF_ZERO
        tag = []
        taggedSentense = []
        for i in range(tagSize * tagSize):
           if prob[-1][i] > maxProb:
               maxProb = prob[-1][i]
               maxEnd = i

        lastTag = maxEnd / tagSize
        #endTag = maxEnd % tagSize
        #tag.append(endTag)
        tag.append(lastTag)
        taggedSentense = []
        for t in range(len(back)):
           currentTag = lastTag
           lastTag = back[-t-1][maxEnd]
           maxEnd = lastTag * tagSize + currentTag
           tag.append(lastTag)

        tag.pop(-1)
        tag.pop(-1)
        tag.reverse()

        sentense.pop(-1)

        for i in range(len(sentense)):
           taggedSentense.append(sentense[i])
           taggedSentense.append("/")
           taggedSentense.append(tL[tag[i]])
           taggedSentense.append(" ")
        taggedSentense.pop(-1)
        taggedSentense.append('\r\n')
        tagged.append(''.join(taggedSentense))

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training,backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training,backoff = bigram_tagger)

    for sentence in brown_dev_words:
        sentence_tag = trigram_tagger.tag(sentence)
        sents = ""
        for tup in sentence_tag:
            word = tup[0]+"/"+tup[1]+" "
            sents += word
        sents +='\r\n'
        tagged.append(sents)
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
