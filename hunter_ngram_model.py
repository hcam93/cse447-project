# Hunter Camfield
# NLP CSE 447 UW
# hcam93@uw.edu

import math

START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"
UNK = "UNK"
THRESHOLD = 3

test_path = "/Users/huntercamfield/Documents/UW_Classes/CSE447/HW3/A3/1b_benchmark.test.tokens"
train_path = "/Users/huntercamfield/Documents/UW_Classes/CSE447/HW3/A3/1b_benchmark.train.tokens"
dev_path = "/Users/huntercamfield/Documents/UW_Classes/CSE447/HW3/A3/1b_benchmark.dev.tokens"


def tokenize_data(sentence_array):
    tokenize_list = []
    word_freq_dict = dict({})
    for sentence in sentence_array:
        sentence = START_TOKEN + " " + sentence + " " + STOP_TOKEN
        for word in sentence.split(" "):
            tokenize_list.append(word)
            if word in word_freq_dict:
                word_freq_dict[word] = word_freq_dict[word] + 1
            else:
                word_freq_dict[word] = 1
    return tokenize_list, word_freq_dict
        

def get_sentences(path):
    line_list = []
    with open(path, encoding = "ISO-8859-1") as rd:
        lines = rd.readlines()
        for line in lines:
            line = line[:len(line) -1] #remove new line char
            line_list.append(line)
    rd.close()
    return line_list


def list_to_dict(list):
    temp_dict = dict({})
    for item in list:
        if item in temp_dict:
            temp_dict[item] = temp_dict[item] + 1
        else:
            temp_dict[item] = 1
    return temp_dict


def unigram_corpus_length(token_list):
    temp = list_to_dict(token_list)
    if START_TOKEN in temp:
        return sum(list(temp.values())) - temp[START_TOKEN]
    else:
        return sum(list(temp.values()))

def unigram_unique_types(token_dict):
    return len(list(token_dict.keys())) - 1

 
def unigram_word_probability(current_word, token_dict: dict, lidstone_smoothing: bool):
    numerator = token_dict[current_word] if current_word in token_dict else token_dict[UNK]
    denominator = 1622905 #freq word count for unigram.
    lidstone_factor = 0.5
    if lidstone_smoothing:
        numerator += lidstone_factor
        denominator += (lidstone_factor * unigram_unique_types(token_dict))
    probability = float(numerator) / float(denominator)
    return probability


def unigram_perplexity(token_dict: dict, token_list: list, lidstone_smoothing: bool):
    word_perplexity = 0
    for word in token_list:
        propability = unigram_word_probability(word, token_dict, lidstone_smoothing)
        word_perplexity += math.log(propability, 2)
    words_len = unigram_corpus_length(token_list)
    perplexity = math.pow(2, (-1 / words_len) * word_perplexity)
    return perplexity


def biTrigram_corpus_length(bigram_tokens):
    return sum(bigram_tokens.values())


def biTrigram_unique_types(bigram_tokens):
    return len(list(bigram_tokens.keys()))


def make_bigram_dict(token_list, unigram_freq):
    bigram_freq = dict({})
    for i in range(1, len(token_list)):
        token_1 = token_list[i-1]
        token_2 = token_list[i]
        if token_1 not in unigram_freq:
            token_1 = UNK
        if token_2 not in unigram_freq:
            token_2 = UNK
        if (token_1, token_2) in bigram_freq:
            bigram_freq[(token_1, token_2)] =  bigram_freq[(token_1, token_2)] + 1
        else:
            bigram_freq[(token_1, token_2)] = 1
        
    return bigram_freq, biTrigram_corpus_length(bigram_freq), biTrigram_unique_types(bigram_freq)
    


def bigram_pair_probability(bigram_pair, bigram_freq, unigram_freq, lidstone):
    lidstone_factor = 0.5
    token_1, token_2 = bigram_pair
    if token_1 not in unigram_freq:
        token_1 = UNK
    if token_2 not in unigram_freq:
        token_2 = UNK
    numerator = bigram_freq[(token_1, token_2)] if (token_1, token_2) in bigram_freq else 0
    denominator = unigram_freq[token_1] if token_1 in unigram_freq else unigram_freq[UNK]
    if lidstone:
        numerator += lidstone_factor
        denominator += (lidstone_factor * unigram_unique_types(unigram_freq))
    probability = float(numerator) / float(denominator)
    return probability


def bigram_perplexity(token_list, bigram_freq, token_freq, lidstone):
    perplexity = 0
    for i in range(1, len(token_list)):
        token_1 = token_list[i-1]
        token_2 = token_list[i]
        bi_prob = bigram_pair_probability((token_1, token_2), bigram_freq, token_freq, lidstone)
        try:
            perplexity += math.log(bi_prob, 2)
        except:
            return 'inf'
    
    word_count = unigram_corpus_length(token_list)
    perplexity = math.pow(2, (-1 / word_count) * perplexity)
    return perplexity


def make_trigram_dict(token_freq, token_list):
    trigram_freq = dict({})
    for i in range(2, len(token_list)):
        token_1 = token_list[i-2]
        token_2 = token_list[i-1]
        token_3 = token_list[i]
        if token_1 not in token_freq:
            token_1 = UNK
        if token_2 not in token_freq:
            token_2 = UNK
        if token_3 not in token_freq:
            token_3 = UNK
        if (token_1, token_2, token_3) in trigram_freq:
            trigram_freq[(token_1, token_2, token_3)] = trigram_freq[(token_1, token_2, token_3)] + 1
        else:
            trigram_freq[(token_1, token_2, token_3)] = 1
    return trigram_freq, biTrigram_corpus_length(trigram_freq), biTrigram_unique_types(trigram_freq)


def trigram_probability(trigram_pair, trigram_freq, bigram_freq, token_freq, smoothing):
    token_1, token_2, token_3 = trigram_pair
    lidstone_factor = 0.5
    numerator = 0
    denominator = 0
    if token_1 not in token_freq:
        token_1 = UNK
    if token_2 not in token_freq:
        token_2 = UNK
    if token_3 not in token_freq:
        token_3 = UNK
    if (token_1, token_2, token_3) in trigram_freq:
        numerator = trigram_freq[(token_1, token_2, token_3)]
    if (token_1, token_2) in bigram_freq:
        denominator = bigram_freq[(token_1, token_2)]
    if smoothing:
        numerator += lidstone_factor
        denominator += (lidstone_factor * unigram_unique_types(token_freq))
    if denominator == 0:
        return 'inf'
    if numerator == 0:
        return 0
    
    probability = float(numerator) / float(denominator)
    return probability


def linear_interpolation_prob(trigram_pair, trigram_freq, bigram_freq, token_freq, linear_interpol_factors, lidstone):
    lam1, lam2, lam3 = linear_interpol_factors
    token1, token2, token3 = trigram_pair
    tri_prob = trigram_probability(trigram_pair, trigram_freq, bigram_freq, token_freq, lidstone) * lam3
    bi_prob = bigram_pair_probability((token3, token2), bigram_freq, token_freq, lidstone) * lam2
    un_prob = unigram_word_probability(token3, token_freq, lidstone) * lam1
    return un_prob + bi_prob + tri_prob


def trigram_perplexity(token_list, token_dict, interpol_smoothing: bool, trigram_freq, bigram_freq, interpol_fact, lidstone_smooth):
    tri_perplex = 0
    for i in range(2, len(token_list)):
        token1 = token_list[i-2]
        token2 = token_list[i-1]
        token3 = token_list[i]
        probability = 0
        if interpol_smoothing:
            probability = linear_interpolation_prob((token1, token2, token3), trigram_freq, bigram_freq, token_dict, interpol_fact, lidstone_smooth)
        else:
            probability = trigram_probability((token1, token2, token3), trigram_freq, bigram_freq, token_dict, lidstone_smooth)
        try:
            tri_perplex += math.log(probability, 2)
        except:
            return "inf"
    return math.pow(2, (-1 / unigram_corpus_length(token_list))  * tri_perplex)

def count_UNK_and_remove(freq_dict: dict, unk_threshold):
    count = 0
    remove_list = []
    for word in freq_dict.keys():
        if freq_dict[word] < unk_threshold:
            remove_list.append(word)
            count += 1
    for word in remove_list:
        del freq_dict[word]
    freq_dict[UNK] = count
    return freq_dict

train_token_list, train_token_freq = tokenize_data(get_sentences(train_path))
train_token_freq = count_UNK_and_remove(train_token_freq, THRESHOLD)
test_token_list, test_token_freq = tokenize_data(get_sentences(test_path))
dev_tokens_list, dev_token_freq = tokenize_data(get_sentences(dev_path))
bigram_freq, bigram_corpus_length, bigram_unique_types = make_bigram_dict(train_token_list, train_token_freq)
trigram_freq, trigram_corpus_length, trigram_unique_types = make_trigram_dict(train_token_freq, train_token_list)

print("Unsmooth perplexity:")

u_train_perp = unigram_perplexity(train_token_freq, train_token_list, False)
u_dev_perp = unigram_perplexity(train_token_freq, dev_tokens_list, False)
u_test_perp = unigram_perplexity(train_token_freq, test_token_list, False)

print("Unigram corpus length: "+str(unigram_corpus_length(train_token_list)))
print("Unigram unique length: "+str(unigram_corpus_length(train_token_freq)))

print("bigram corpus length: "+ str(biTrigram_corpus_length(bigram_freq)))
print("bigram unique length: "+ str(biTrigram_unique_types(bigram_freq)))

print("trigram corpus length: "+ str(biTrigram_corpus_length(trigram_freq)))
print("trigram unique length: "+ str(biTrigram_unique_types(trigram_freq)))

print("Unigram perplexity on training data: " + str(u_train_perp))
print("Unigram perplexity on dev data:      " + str(u_dev_perp))
print("Unigram perplexity on test data:     " + str(u_test_perp))


bi_train_perp = bigram_perplexity(train_token_list, bigram_freq, train_token_freq, False)
bi_dev_perp = bigram_perplexity(dev_tokens_list, bigram_freq, train_token_freq, False)
bi_test_perp = bigram_perplexity(test_token_list, bigram_freq, train_token_freq, False)

print("Bigram perplexity on training data: " + str(bi_train_perp))
print("Bigram perplexity on dev data:      " + str(bi_dev_perp))
print("Bigram perplexity on test data:     " + str(bi_test_perp))

tri_train_perp =  trigram_perplexity(train_token_list, train_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), False)
tri_dev_perp = trigram_perplexity(dev_tokens_list, dev_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), False)
tri_test_perp = trigram_perplexity(test_token_list, test_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), False)

print("Trigram perplexity on training data: " + str(tri_train_perp))
print("Trigram  perplexity on dev data:     " + str(tri_dev_perp))
print("Trigram  perplexity on test data:    " + str(tri_test_perp))

u_train_smooth_perp = unigram_perplexity(train_token_freq, train_token_list, True)
u_dev_smooth_perp = unigram_perplexity(train_token_freq, dev_tokens_list, True)
u_test_smooth_perp = unigram_perplexity(train_token_freq, test_token_list, True)

print("Smoothed perplexities")
print("Unigram perplexity on training data with smoothing: " + str(u_train_smooth_perp))
print("Unigram perplexity on dev data with smoothing:      " + str(u_dev_smooth_perp))
print("Unigram perplexity on test data with smoothing:     " + str(u_test_smooth_perp))

bi_train_smooth_perp = bigram_perplexity(train_token_list, bigram_freq, train_token_freq, True)
bi_dev_smooth_perp = bigram_perplexity(dev_tokens_list, bigram_freq, train_token_freq, True)
bi_test_smooth_perp = bigram_perplexity(train_token_list, bigram_freq, train_token_freq, True)

print("Bigram perplexity on training data with smoothing: " + str(bi_train_smooth_perp))
print("Bigram perplexity on dev data with smoothing:      " + str(bi_dev_smooth_perp))
print("Bigram perplexity on test data with smoothing:     " + str(bi_test_smooth_perp))

tri_train_smooth_perp =  trigram_perplexity(train_token_list, train_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), True)
tri_dev_smooth_perp = trigram_perplexity(dev_tokens_list, train_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), True)
tri_test_smooth_perp = trigram_perplexity(test_token_list, train_token_freq, False, trigram_freq, bigram_freq, (1/3, 1/3, 1/3), True)


print("Trigram perplexity on training data with smoothing: " + str(tri_train_smooth_perp))
print("Trigram perplexity on dev data with smoothing:      " + str(tri_dev_smooth_perp))
print("Trigram perplexity on test data with smoothing:     " + str(tri_test_smooth_perp))


print("Perplexity with linear interpolation")
hyperparams = [
    (0.01, 0.03, 0.6),
    (0.01, 0.05, 0.94),
    (0.06, 0.09, 0.85),
    (0.2, 0.24, 0.56),
    (0.21, 0.33, 0.46),
    (0.34, 0.34, 0.32)
]

for param in hyperparams:
    # train = trigram_perplexity(train_token_list, train_token_freq, True, trigram_freq, bigram_freq, param, False )
    dev = trigram_perplexity(dev_tokens_list, train_token_freq, True, trigram_freq, bigram_freq, param, False )
    test = trigram_perplexity(test_token_list, train_token_freq, True, trigram_freq, bigram_freq, param, False )
    # print("training : " + str(train) + ", params: " + str(param))
    print("dev : " + str(dev) + ", params: " + str(param))
    print("test : " + str(test) + ", params: " + str(param))


print("Half the training data")
half_tokens_list = train_token_list[:int(len(train_token_list)/2)]
half_tokens_freq = list_to_dict(half_tokens_list)
half_tokens_freq = count_UNK_and_remove(half_tokens_freq, 3)
tri_half_tokens, a, b = make_trigram_dict(half_tokens_freq, half_tokens_list)
dev_tri = trigram_perplexity(dev_tokens_list, tri_half_tokens, True, trigram_freq, bigram_freq, (.1, .3, .6), False)
test_tri = trigram_perplexity(test_token_list, tri_half_tokens, True, trigram_freq, bigram_freq, (.1, .3, .6), False)
print("Halfing the training data with dev:  " + str(dev_tri))
print("Halfing the training data with test: " + str(dev_tri))


print("Change unk to 5 occurances")
train_list, train_freq = tokenize_data(get_sentences(train_path))
train_freq = count_UNK_and_remove(train_freq, 5)
train_u = unigram_perplexity(train_freq, train_list, False)
test_u = unigram_perplexity(train_freq, test_token_list, False)
dev_u = unigram_perplexity(train_freq, dev_tokens_list, False)
print("UNK < 5 unigram: ")
print("train: " + str(train_u))
print("test : " + str(test_u))
print("dev  : " + str(dev_u))