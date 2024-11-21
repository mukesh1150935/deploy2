#ble_score.py
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

def read_references():
    """
    预料的refetences计算
    :return: [ [['word','word'],['word','word']]   ]
    """
    result = []
    r_sentences = []

    from setting import BLEU_REFERENCES
    f = open(BLEU_REFERENCES,'r',encoding='utf-8')
    sentences = f.readlines()
    for s in sentences:
        references = []
        references.append(s.strip().split(' '))
        result.append(references)
    f.close()
    return result

def read_candidates():
    result = []
    from setting import BLEU_CANDIDATE
    file = open(BLEU_CANDIDATE,'r',encoding='utf-8')
    sentences = file.readlines()
    for s in sentences:
        result.append(s.strip().split(' '))
    file.close()
    return result


if __name__ == '__main__':

    references = read_references()
    candidates = read_candidates()
    score = corpus_bleu(references, candidates,weights=(1,0,0,0))
    print("BLEU Score for 1:0:0:0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,0.2,0,0))
    print("BLEU Score for 1,0.2,0,0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,0.5,0.2,0.1))
    print("BLEU Score for 1,0.5,0.2,0.1 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,0.2,0.2,0.2))
    print("BLEU Score for 1,0.2,0.2,0.2 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(0.8,0.5,0.3,0))
    print("BLEU Score for 0.8,0.5,0.3,0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,1,0,0))
    print("BLEU Score for 1:1:0:0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,1,1,0))
    print("BLEU Score for 1:1:1:0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(1,1,1,1))
    print("BLEU Score for 1:1:1:1 weitage is")
    print(score*100)
    
    score = corpus_bleu(references, candidates,weights=(0,1,0,0))
    print("BLEU Score for 0,1,0,0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(0,0,1,0))
    print("BLEU Score for 0,0,1,0 weitage is")
    print(score*100)
    score = corpus_bleu(references, candidates,weights=(0,0,0,1))
    print("BLEU Score for 0,0,0,1 weitage is")
    print(score*100)
    