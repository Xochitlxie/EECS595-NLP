import A
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from nltk import word_tokenize
from nltk.corpus import cess_esp
from nltk.corpus import cess_cat
from nltk.data import load
from sklearn import svm
import nltk
from nltk import UnigramTagger as ut
import operator
import math
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#english_stop = stopwords.words("english")
#spanish_stop = stopwords.words("spanish")
#english_stem = SnowballStemmer("english")
#spanish_stem = SnowballStemmer("spanish")

# You might change the window size
window_size = 25

def b1_base(data,language):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    s = []
    for context in data:
        left_token = word_tokenize(context[1])
        right_token = word_tokenize(context[3])
        if len(left_token) < window_size:
            s += left_token
        else:
            s += left_token[-window_size:]
        if len(right_token) < window_size:
            s += right_token
        else:
            s += right_token[:window_size]

    vectors = {}
    labels = {}

    for instance in data:
        left = instance[1]
        right = instance[3]

        """
        # remove punctuation
        if language == 'English':
            left_stop = ' '.join([word for word in left.split() if word not in english_stop])
            right_stop = ' '.join([word for word in right.split() if word not in english_stop])
            temple_left = word_tokenize(left_stop)
            temple_right = word_tokenize(right_stop)
            left_pun = []
            right_pun = []
            for i in temple_left:
                if i in string.punctuation:
                    left_pun.append(i)
            for j in temple_right:
                if j in string.punctuation:
                    right_pun.append(j)
            left_token = []
            right_token = []
            for i in left_pun:
                left_token.append(english_stem.stem(i))
            for j in right_pun:
                right_token.append(english_stem.stem(i))


        elif language == 'Spanish':
            left_stop = ' '.join([word for word in left.split() if word not in spanish_stop])
            right_stop = ' '.join([word for word in right.split() if word not in spanish_stop])
            temple_left = word_tokenize(left_stop)
            temple_right = word_tokenize(right_stop)
            left_pun = []
            right_pun = []
            for i in temple_left:
                if i in string.punctuation:
                    left_pun.append(i)
            for j in temple_right:
                if j in string.punctuation:
                    right_pun.append(j)
            left_token = []
            right_token = []
            for i in left_pun:
                left_token.append(spanish_stem.stem(i))
            for j in right_pun:
                right_token.append(spanish_stem.stem(j))

        elif language == 'Catalan':
            temple_left = word_tokenize(left)
            temple_right = word_tokenize(right)
            left_token = []
            right_token = []
            for i in temple_left:
                if i in string.punctuation:
                    left_token.append(i)
            for j in temple_right:
                if j in string.punctuation:
                    right_token.append(j)
        # remove punctuation
        """
        left_token = word_tokenize(left)
        right_token = word_tokenize(right)
        vector_words = left_token + right_token
        instance_id = instance[0]
        sense_id = instance[4]
        wordset = set(s)
        count_id = {}
        for word in wordset:
            count_id[word] = count_id.get(word,0) + vector_words.count(word)
        vectors[instance_id] = count_id
        labels[instance_id] = sense_id

    # implement your code here

    return vectors, labels

def extra_wn(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return wn_dict: A dictionary with the following structure
        {
        instanced_id : [hypernyms,hyponyms,hypernyms]

    '''
    wn_dict = {}
    for context in data:
        instance_id = context[0]
        head = context[2]
        wn_sample = wn.synsets(head)
        if len(wn_sample)==0:
            wn_dict[instance_id] = [str(0),str(0),str(0),str(0)]
        else:
            wn_word = wn_sample[0]
            if len(wn_word.hypernyms())==0:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(0)]
            else:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(wn_word.hypernyms()[0])]

            if len(wn_word.hyponyms())==0:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(0)]
            else:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(wn_word.hyponyms()[0])]

            if len(wn_word.member_holonyms())==0:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(0)]
            else:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(wn_word.member_holonyms()[0])]

            if len(wn_word.root_hypernyms())==0:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(0)]
            else:
                wn_dict[instance_id] = wn_dict.get(instance_id,[]) + [str(wn_word.root_hypernyms()[0])]

    return wn_dict


def extra_english(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: extraW: A dictionary with the following structure
            { (instance_id): [w-2,w-1,w0,w1,w2],
            ...

    '''
    dict_w = {}
    for context in data:
        # remove punctuations
        # left = re.sub('[%s]' % re.escape(string.punctuation), '', context[1])
        # right = re.sub('[%s]' % re.escape(string.punctuation), '', context[3])
        left_token = word_tokenize(context[1])
        right_token = word_tokenize(context[3])
        if len(left_token) == 0:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0),str(0)]
        elif len(left_token) == 1:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0)] + left_token
        elif len(left_token) == 2:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0)] + left_token
        else:
            dict_w[context[0]] = dict_w.get(context[0],[]) + left_token[-3:]
        dict_w[context[0]] = dict_w.get(context[0],[]) + [context[2]]
        if len(right_token) == 0:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0),str(0)]
        elif len(right_token) == 1:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0)] + right_token
        elif len(right_token) == 2:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0)] + right_token
        else:
            dict_w[context[0]] = dict_w.get(context[0],[]) + right_token[:3]

    return dict_w



def countFeature(features, data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    '''
    count = {}
    for instance in data:
        left_words = word_tokenize(instance[1])[-window_size:]
        right_words = word_tokenize(instance[3])[0:window_size]
        instance_id = instance[0]
        count[instance_id] = {}
        for word in left_words + right_words:
            if word in features:
                count[instance_id][word] = count[instance_id].get(word, 0) + 1
    return count

def getBestWords(data, n):
    '''
    Get best words according to

    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param n: Get n best words
    :return: a set containing selected features
    '''
    senseCount = {}
    cCount = {}

    for instance in data:
        left_words = word_tokenize(instance[1])
        right_words = word_tokenize(instance[3])
        left_words = left_words[-window_size:]
        right_words = right_words[0:window_size]
        sense_id = instance[4]

        for word in left_words + right_words:
            cCount[word] = cCount.get(word, 0) + 1
            senseCount[sense_id] = senseCount.get(sense_id, {})
            senseCount[sense_id][word] = senseCount[sense_id].get(word, 0) + 1

    feature_set = set()

    score = {}
    for sense_id in senseCount.keys():
        score[sense_id] = {}
        for word in senseCount[sense_id].keys():
            if senseCount[sense_id][word] == cCount[word]:
                score[sense_id][word] = float('Inf')
            else:
                p = 1.0 * senseCount[sense_id][word] / cCount[word]
                score[sense_id][word] = math.log(p/(1-p))

    for sense_id in score.keys():
        features = sorted(score[sense_id].items(), key=operator.itemgetter(1), reverse = True)
        for feature in features[0 : n]:
            feature_set.add(feature[0])

    return feature_set

def extra_w(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: extraW: A dictionary with the following structure
            { (instance_id): [w-2,w-1,w0,w1,w2],
            ...

    '''
    dict_w = {}
    for context in data:
        # remove punctuations
        # left = re.sub('[%s]' % re.escape(string.punctuation), '', context[1])
        # right = re.sub('[%s]' % re.escape(string.punctuation), '', context[3])
        left_token = word_tokenize(context[1])
        right_token = word_tokenize(context[3])
        if len(left_token) == 0:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0)]
        elif len(left_token) == 1:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0)] + left_token
        # elif len(left_token) == 2:
        #dict_w[context[0]] = dict_w.get(context[0],[]) + [0] + left_token
        else:
            dict_w[context[0]] = dict_w.get(context[0],[]) + left_token[-2:]
        dict_w[context[0]] = dict_w.get(context[0],[]) + [context[2]]
        if len(right_token) == 0:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0),str(0)]
        elif len(right_token) == 1:
            dict_w[context[0]] = dict_w.get(context[0],[]) + [str(0)] + right_token
        # elif len(left_token) == 2:
        #dict_w[context[0]] = dict_w.get(context[0],[]) + [0] + right_token
        else:
            dict_w[context[0]] = dict_w.get(context[0],[]) + right_token[:2]

    return dict_w

def extraPOS(language,extraW,tagger):
    '''
    :param language: which languge this function apply (string)
    :param extraW: A dictionary with the following structure
            { (instance_id): [w-2,w-1,w0,w1,w2],
            ...
            }
    :return: extraW: A dictionary with the following structure
            English
            { (instance_id): [POS-2,POS-1,POS,POS1,POS2],
            ...
            }
            #Spanish and Catalan
            #{ (instance_id): [POS-1,POS,POS1],
            #...
            #}
    '''
    extraPOS = {}
    if language == 'English':
        for inId in extraW.keys():
            wordPOS = tagger.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    if language == 'Catalan':
        for inId in extraW.keys():
            wordPOS = tagger.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    if language == 'Spanish':
        for inId in extraW.keys():
            wordPOS = tagger.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    return extraPOS

# B.1.a,b,c,d
def extract_features(data,language,tagger):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
                        [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}
    features, labels = b1_base(data,language)


    # best
    if language == 'English':
        words = extra_english(data)
    else:
        words = extra_w(data)

    POS = extraPOS(language,words,tagger)
    for k in POS.keys():
        for i in range(len(words[k])):
            if language == 'English':
                features[k]['w-3'] = words[k][0]
                features[k]['w-2'] = words[k][1]
                features[k]['w-1'] = words[k][2]
                features[k]['w0'] = words[k][3]
                features[k]['w1'] = words[k][4]
                features[k]['w2'] = words[k][5]
                features[k]['w3'] = words[k][6]
            else:
                features[k]['w-2'] = words[k][0]
                features[k]['w-1'] = words[k][1]
                features[k]['w0'] = words[k][2]
                features[k]['w1'] = words[k][3]
                features[k]['w2'] = words[k][4]
        for i in range(len(POS[k])):
            if language == 'English':
                features[k]['POS-3'] = POS[k][0]
                features[k]['POS-2'] = POS[k][1]
                features[k]['POS-1'] = POS[k][2]
                features[k]['POS0'] = POS[k][3]
                features[k]['POS1'] = POS[k][4]
                features[k]['POS2'] = POS[k][5]
                features[k]['POS3'] = POS[k][6]
            else:
                features[k]['POS-2'] = POS[k][0]
                features[k]['POS-1'] = POS[k][1]
                features[k]['POS0'] = POS[k][2]
                features[k]['POS1'] = POS[k][3]
                features[k]['POS2'] = POS[k][4]
    """
    #B1.a
    words = extra_w(data)
    POS = extraPOS(language,words,tagger)
    for k in POS.keys():
        for i in range(len(words[k])):
            features[k]['w-2'] = words[k][0]
            features[k]['w-1'] = words[k][1]
            features[k]['w0'] = words[k][2]
            features[k]['w1'] = words[k][3]
            features[k]['w2'] = words[k][4]
        for i in range(len(POS[k])):
            features[k]['POS-2'] = POS[k][0]
            features[k]['POS-1'] = POS[k][1]
            features[k]['POS0'] = POS[k][2]
            features[k]['POS1'] = POS[k][3]
            features[k]['POS2'] = POS[k][4]


    # B1.b

    # B1.c
    #feature_list = getBestWords(data, 6)
    # B1.c

    # B1.d
    wn_dict = extra_wn(data)


    for k in wn_dict.keys():
        for i in range(len(wn_dict[k])):
            features[k]['hypernyms'] = wn_dict[k][0]
            features[k]['hyponyms'] = wn_dict[k][1]
            features[k]['member_holonyms'] = wn_dict[k][2]
            features[k]['root_hypernyms'] = wn_dict[k][3]

    """
    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''
    """
    x_train_mat = []
    x_test_mat = []
    y_train_list = []

    train_id_list = X_train.keys()
    test_id_list = X_test.keys()

    for i in train_id_list:
        x_train_mat.append(X_train[i])
        y_train_list.append(y_train[i])

    y_train_list = np.asarray(y_train_list)

    for i in test_id_list:
        x_test_mat.append(X_test[i])

    selector=SelectKBest(chi2, k=20)


    x_train_new = selector.fit_transform(x_train_mat, y_train_list)
    x_test_new = selector.transform(x_test_mat)


    for i in range(len(train_id_list)):
        X_train[train_id_list[i]] = x_train_new[i]
    for i in range(len(test_id_list)):
        X_test[test_id_list[i]] = x_test_new[i]



    """
    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []
    # linear support vector mechines
    svm_clf = svm.LinearSVC()
    svm_data = []
    svm_target = []
    svm_test = []
    svm_testid = []
    for lex in X_train.keys():
        svm_data.append(X_train[lex])
        svm_target.append(y_train[lex])
    for lexi in X_test.keys():
        svm_test.append (X_test[lexi])
        svm_testid.append(lexi)
    svm_clf.fit(svm_data,svm_target)
    svm_label = svm_clf.predict(svm_test)
    for i in range(len(svm_testid)):
        results.append((svm_testid[i],svm_label[i]))

    # implement your code here

    return results

# run part B
def run(train, test, language, answer):
    results = {}
    if language == 'English':
        _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tagger = load(_POS_TAGGER)
    elif language == 'Spanish':
        tagger = ut(cess_esp.tagged_sents())
    elif language == 'Catalan':
        tagger  = ut(cess_cat.tagged_sents())

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt],language,tagger)
        test_features, _ = extract_features(test[lexelt],language,tagger)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)
    """
    B1.c
    for lexelt in train:
        features = getBestWords(train[lexelt], 30)
        train_features = countFeature(features, train[lexelt])
        _, y_train = extract_features(train[lexelt], language)
        test_features = countFeature(features, test[lexelt])

        X_train, X_test = vectorize(train_features, test_features)
        results[lexelt] = classify(X_train, X_test, y_train)
    B1.c
    """
    A.print_results(results, answer)




