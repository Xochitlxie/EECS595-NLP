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

tagger_cat  = ut(cess_cat.tagged_sents())
tagger_esp = ut(cess_esp.tagged_sents())
# You might change the window size
window_size = 15

def b1_base(data):
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
        # remove punctuation
        #left = re.sub('[%s]' % re.escape(string.punctuation), '', instance[1])
        #right = re.sub('[%s]' % re.escape(string.punctuation), '', instance[3])
        # remove punctuation
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

    return vectors, labels

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

def extraPOS(language,extraW):
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
        _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tagger = load(_POS_TAGGER)
        for inId in extraW.keys():
            wordPOS = tagger.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    if language == 'Catalan':
        for inId in extraW.keys():
            wordPOS = tagger_cat.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    if language == 'Spanish':
        for inId in extraW.keys():
            wordPOS = tagger_esp.tag(extraW[inId])
            for i in wordPOS:
                extraPOS[inId] = extraW.get(inId,[])+[i[1]]

    return extraPOS


# B.1.a,b,c,d
def extract_features(data,language):
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
    features, labels = b1_base(data)

    #B1.a
    words = extra_w(data)
    POS = extraPOS(language,words)
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
    # implement your code here

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

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt],language)
        test_features, _ = extract_features(test[lexelt],language)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)




