import nltk
import A
from nltk.align import AlignedSent
import time
import math

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignment = []
        m = len(align_sent.mots)
        l = len(align_sent.words)
        f = align_sent.mots
        e = align_sent.words
        for j in range(0,l):
            prob = {}
            for i in range(0,m):
                prob[(i,j)] =(self.q[(j,i,l,m)] * self.t[(f[i],e[j])])*(self.q[(i , j , m ,l)] * self.t[(e[j],f[i])])
            sorted_prob = sorted(prob.iteritems(), key=lambda d:d[1], reverse = True)
            t = sorted_prob[0][0]
            alignment.append(str(t[1])+'-'+str(t[0]))
        str_ali = alignment[0] + ' '
        for a in range(1,len(alignment)-1):
            str_ali += alignment[a] + ' '
        str_ali += alignment[len(alignment)-1]

        Ali = AlignedSent(e,f,str_ali)

        return Ali

    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        t = {}
        q = {}
        # initial t, q
        ne = {}
        ne_count = {}
        nf = {}
        nf_count = {}
        for i in aligned_sents:
            for words in i.words:
                ne[words] = ne.get(words,[]) + i.mots
            for motsWords in i.mots:
                nf[motsWords] = nf.get(motsWords,[]) + i.words
        for k in ne:
            ne_count[k] = ne_count.get(k,0.0) + 1.0 * len(set(ne[k]))
        for z in nf:
            nf_count[z] = nf_count.get(z,0.0) + 1.0 * len(set(nf[z]))

        for sen in aligned_sents:
            for e in sen.words:
                for f in sen.mots:
                    t[(f,e)] = t.get((f,e),0.0)
                    if t[(f,e)] == 0.0:
                        t[(f,e)] += 1.0 / ne_count[e]
                    t[(e,f)] = t.get((e,f),0.0)
                    if t[(e,f)] == 0.0:
                        t[(e,f)] += 1.0 / nf_count[f]
            m = len(sen.mots)
            l = len(sen.words)
            for i in range(0,m):
                for j in range(0,l):
                    q[(j,i,l,m)] = q.get((j,i,l,m),0.0)
                    if q[(j,i,l,m)] == 0.0:
                        q[(j,i,l,m)] += 1.0 / l
                    q[(i,j,m,l)] = q.get((i,j,m,l),0.0)
                    if q[(i,j,m,l)] == 0.0:
                        q[(i,j,m,l)] += 1.0 / m
        # iteration and training
        for s in range(num_iters):
            c = {}
            for sen in aligned_sents:
                m = len(sen.mots)
                l = len(sen.words)
                """
                for i in range(0,m):
                    for detla_j in range(0,l):
                        sum_detla1 += q[('j',detla_j,'i',i,l,m)] * t[(sen.mots[i],sen.words[detla_j])]
                        sum_detla2 += q[('i',i,'j',detla_j,m,l)] * t[(sen.words[detla_j],sen.mots[i])]
                """
                for i in range(0,m):
                    sum_detla1 = 0.0
                    sum_detla2 = 0.0
                    for detla_j in range(0,l):
                        sum_detla1 += q[(detla_j,i,l,m)] * t[(sen.mots[i],sen.words[detla_j])]
                        sum_detla2 += q[(i,detla_j,m,l)] * t[(sen.words[detla_j],sen.mots[i])]
                    for j in range(0,l):
                        fi = sen.mots[i]
                        ej = sen.words[j]
                        avg_detla1 = (q[(j,i,l,m)] * t[(fi,ej)]) / sum_detla1
                        avg_detla2 = (q[(i,j,m,l)] * t[(ej,fi)]) / sum_detla2
                        avg_detla = (avg_detla1 + avg_detla2) / 2.0
                        c[(ej,fi)] = c.get((ej,fi),0.0) + avg_detla
                        c[(fi,ej)] = c.get((fi,ej),0.0) + avg_detla
                        c[ej] = c.get(ej,0.0) + avg_detla
                        c[fi] = c.get(fi,0.0) + avg_detla
                        c[(j,i,l,m)] = c.get((j,i,l,m),0.0) + avg_detla
                        c[(i,j,m,l)] = c.get((i,j,m,l),0.0) + avg_detla
                        c[(i,l,m)] = c.get((i,l,m),0.0) + avg_detla
                        c[(j,m,l)] = c.get((j,m,l),0.0) + avg_detla
            for t_tuple in t:
                t[t_tuple] = c[(t_tuple[1],t_tuple[0])] / c[t_tuple[1]]
            for q_tuple in q:
                q[q_tuple] = c[q_tuple] / c[(q_tuple[1],q_tuple[2],q_tuple[3])]
        return (t,q)

def main(aligned_sents):
    time.clock()
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
    print "Part B time: " + str(time.clock()) + ' sec'