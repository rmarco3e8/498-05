import nltk
import re
from nltk.corpus import wordnet
from nltk.sentiment.util import mark_negation

def handle_negation(sentence):
    """
    Adds negative tag to every word between appearance of negation term and the next punctuation mark or next line

    """
    def add_space(matched_obj):
        return " " + matched_obj.group(0)

    # add a period upon new line if new line does not already end with punctuation and is not followed by another new line
    sentence = re.sub(r"(?<![[^\w\s']])\n(?!\n)",". ",sentence)

    # add space before punctuation (not apostrophe's to avoid contraction issues)
    sentence = re.sub(r"[^\w\s']", add_space , sentence)

    # mark negation and put string back together
    sentence = " ".join(mark_negation(sentence.split()))

    return sentence 

def lemmatize(word_list):

    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Tag part of speech if adjective, noun, verb, adverb
    def pos_tagger(nltk_tag): 
        if nltk_tag.startswith('J'): 
            return wordnet.ADJ 
        elif nltk_tag.startswith('V'): 
            return wordnet.VERB 
        elif nltk_tag.startswith('N'): 
            return wordnet.NOUN 
        elif nltk_tag.startswith('R'): 
            return wordnet.ADV 
        else:           
            return None

    pos_tagged = nltk.pos_tag(word_list)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_output = [tup[0] if tup[1] is None else lemmatizer.lemmatize(tup[0], tup[1]) for tup in wordnet_tagged]

    return lemmatized_output


if __name__ == '__main__':

    # For testing
    sentence = ['my', 'dog', 'went', 'on', 'a', 'better', 'adventure']
    print(lemmatize(sentence))

    sentence = ['I', 'am', 'playing']
    print(lemmatize(sentence))

    sentence = ['I', 'play']
    print(lemmatize(sentence))

    sentence = ["I'm", 'just', 'tryna', 'flex', 'on', 'these', 'hoes', 'but', 'I', 'can\'t']
    print(lemmatize(sentence))

    sentence = """An old man said to me, wont see another one"""
    print(handle_negation(sentence))

