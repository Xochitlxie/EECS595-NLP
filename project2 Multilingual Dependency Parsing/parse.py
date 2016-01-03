import sys
import random
from providedcode import dataset
from providedcode.dependencygraph import DependencyGraph
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition

for line in sys.stdin:
    sentence = DependencyGraph.from_sentence(line)
    tp = TransitionParser.load(sys.argv[1])
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8')

#sentence = DependencyGraph.from_sentence('Hi, this is a test')
#tp = TransitionParser.load('english.model')
#parsed = tp.parse([sentence])
#print parsed[0].to_conll(10).encode('utf-8')