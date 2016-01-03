from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk
import string
import re

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dict with the following structure:
        {
                        lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
                        ...
        }
    :return: dict s with the following structure:
        {
                        lexelt: [w1,w2,w3, ...],
                        ...
        }

    '''
    s = {}

    # implement your code here
    for key in data.keys():
        for context in data[key]:
            # left = re.sub('[%s]' % re.escape(string.punctuation), '', context[1])
            # right = re.sub('[%s]' % re.escape(string.punctuation), '', context[3])
            left_token = nltk.word_tokenize(context[1])
            right_token = nltk.word_tokenize(context[3])
            if len(left_token) < window_size:
                s[key] = s.get(key,[]) + left_token
            else:
                s[key] = s.get(key,[]) + left_token[-window_size:]
            if len(right_token) < window_size:
                s[key] = s.get(key,[]) + right_token
            else:
                s[key] = s.get(key,[]) + right_token[:window_size]
    return s

# A.1
def vectorize(data, s):
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
    vectors = {}
    labels = {}
    for instance in data:
        left = instance[1]
        right = instance[3]
        #left = re.sub('[%s]' % re.escape(string.punctuation), '', instance[1])
        #right = re.sub('[%s]' % re.escape(string.punctuation), '', instance[3])
        left_token = nltk.word_tokenize(left)
        right_token = nltk.word_tokenize(right)
        vector_words = left_token + right_token
        instance_id = instance[0]
        sense_id = instance[4]
        wordset = set(s)
        count_id = []
        for word in wordset:
            count_id.append(vector_words.count(word))
        vectors[instance_id] = count_id
        labels[instance_id] = sense_id

    # implement your code here

    return vectors, labels

def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

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
        svm_results.append((svm_testid[i],svm_label[i]))


    # k-nearest neighbors KNN
    knn_clf = neighbors.KNeighborsClassifier()
    knn_data = []
    knn_target = []
    knn_test =[]
    knn_testid = []
    for key in X_train.keys():
        knn_data.append(X_train[key])
        knn_target.append(y_train[key])
    for keys in X_test.keys():
        knn_test.append(X_test[keys])
        knn_testid.append(keys)
    knn_clf.fit(knn_data,knn_target)
    knn_label = knn_clf.predict(knn_test)
    for i in range(len(knn_testid)):
        knn_results.append((knn_testid[i],knn_label[i]))

    # implement your code here
    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''
    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results alphabetically by lexelt_item, then on
    # instance_id before printing
    remove_accent = {}
    for lexelt in results.keys():
        listOfTuple = []
        for tuple_k in results[lexelt]:
            listOfTuple.append((replace_accented(unicode(tuple_k[0])),replace_accented(unicode(tuple_k[1]))))
        sorted_tuple = sorted(listOfTuple,key = lambda x : x[0])
        remove_accent[replace_accented(lexelt)] = sorted_tuple

    sorted_rm = sorted(remove_accent.items(), key=lambda d:d[0])

    f = file(output_file,'w')
    for tuple_z in sorted_rm:
        for l in tuple_z[1]:
            line = tuple_z[0] + " " + l[0] + " " + l[1] + '\n'
            f.write(line)
    f.close()


# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)

