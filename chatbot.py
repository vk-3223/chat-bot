import sys
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas
import wikipedia
from sklearn.preprocessing import OneHotEncoder

lemmatizer = WordNetLemmatizer()

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')


topic = input("which topic would you like discuss? ").title()

try:
    text = wikipedia.page(topic).content
except:
    print("This topic not avalible")
    sys.exit()
lemmatizer = WordNetLemmatizer()


  
def lemma(sent):
    stentec_lemma = []
    sentence_token = nltk.word_tokenize(sent.lower())
    pos_tages = nltk.pos_tag(sentence_token)
    for token,pos_tage in zip(sentence_token,pos_tages):
        if pos_tage[1][0].lower() in ['n','v','a','r']:
            lemma = lemmatizer.lemmatize(token,pos_tage[1][0].lower())
            stentec_lemma.append(lemma)
        
    return (stentec_lemma)

def processs(text,question):
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfVectorizer(tokenizer=lemma)
    tf = tv.fit_transform(sentence_tokens)

    df = pandas.DataFrame(tf.toarray(),columns=tv.get_feature_names_out())

    values = cosine_similarity(tf[-1],tf)


    index = values.argsort()[0][-2]


    values_flatt = values.flatten()
    values_flatt.sort()

    coff = values_flatt[-2]


    if coff>0.3:
        return sentence_tokens[index]
    
while True:
    question = input("Hi, What do you want to know?\n")
    output = processs(text,question)
    if output:
        print(output)
    if question=="quiet":
        break
    