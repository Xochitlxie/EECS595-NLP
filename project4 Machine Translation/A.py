import nltk
from nltk.align import IBMModel1
from nltk.align import IBMModel2
import time

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm =  IBMModel1(aligned_sents,10)
    return ibm

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm = IBMModel2(aligned_sents,10)
    return ibm

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    AER_sum = 0.0
    for i in range(0,n):
        alignment = aligned_sents[i].alignment
        trained_alignment = model.align(aligned_sents[i]).alignment
        AER =  aligned_sents[i].alignment_error_rate(trained_alignment)
        AER_sum += AER
    average_AER = AER_sum / n
    return average_AER

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    outfile = open(file_name,'w')
    list_of_aligned_sents = []
    for i in range(0,20):
        list_of_aligned_sents.append(aligned_sents[i])
    for alignment in list_of_aligned_sents:
        outfile.write(str(alignment.words) + '\n')
        outfile.write(str(alignment.mots) + '\n')
        outfile.write(str(model.align(alignment).alignment) + '\n')
        outfile.write('\n')

def main(aligned_sents):
    time.clock()
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")

    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)

    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
    print "Part A time: " + str(time.clock()) + ' sec'
