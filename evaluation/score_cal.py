import csv
import gleu_score
import f_score
import nltk
import bleu_score

nltk.download('punkt')

#usage: call function score(path_to_eval_file) and it will return bleu_score, f_core, gleu_score in lexicographic order

def score(x='eval_new.csv', beta=1):
    
    predict = []
    ref = []
    temp = []
    gsum = 0
    fsum = 0
    bsum = 0

    with open(x, 'r') as file:
        reader = list(csv.reader(file))

    for i in range(1,len(reader)):
        if i%4 == 1:
            predict.append(reader[i][2][8:])
            if temp != []:
                ref.append(temp)
            temp = []
            temp.append(reader[i][1])

        else:
            temp.append(reader[i][1])

    ref.append(temp)

    for i in range(0,len(ref)):
        gscore = gleu_score.sentence_gleu(ref[i],predict[i])
        gsum += gscore

        #if no beta provided, f1 will be processed as default
        fscore = f_score.f_beta_score(ref[i], predict[i], beta)
        fsum += fscore
        
        bscore = bleu_score.sentence_bleu(ref[i],predict[i])
        bsum += bscore
                                        

    return bsum/len(ref), fsum/len(ref), gsum/len(ref)
    