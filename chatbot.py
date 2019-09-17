'''
CODY - TextBot
A simple NLTK Text Chatbot for the "Tag der Talente" workshop 2019 
based on an example script from Parul Pandey.
'''
import io
import os
import random
import string # (Zur Verarbeitung von Standard Python Strings)
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from termcolor import colored, cprint
import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# Begrüßungen
GREETING_INPUTS = ("hallo", "hi", "grüße", "tach", "was geht", "hey")
GREETING_RESPONSES = ["hi", "hey", "gott zum gruße", "tach", "hallo", "Es freut mich, mit dir sprechen zu dürfen."]

# Beleidigungen
INDIGNITY_INPUTS = ("arsch", "sau", "depp", "doof", "dumm", "kacke")
INDIGNITY_RESPONSES = ["Wir sollten nett zueinander sein.", "Wenn du meinst.", "Überleg mal, was du sagst.", "Das finde ich nicht nett.", "Du solltest sowas nicht sagen", "Ohje, du bist ja ein besonders netter Zeitgenosse..."]

nltk.download('popular', quiet=True) 

# Für den ersten Start, ansonsten auskommentieren
#nltk.download('punkt') 
#nltk.download('wordnet') 


# Corpus einlesen
with open('chatbot_de.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

# Tokenisierung
# sent_tokens konvertiert in Liste von Sätzen
sent_tokens = nltk.sent_tokenize(raw)
# word_tokens konvertiert in Liste von Worten (Wird nicht verwendet.)
word_tokens = nltk.word_tokenize(raw)

# Vorverarbeitung (Preprocessing)
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
def trivia(sentence):
    '''Wenn die Nutzereingabe ien Begrüßung ist, Antwortet der Bot mit einer zufälligen Begrüßung als Antwort, 
    gleiches gilt für Beleidigungen'''
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        if word.lower() in INDIGNITY_INPUTS:
            return random.choice(INDIGNITY_RESPONSES)


# Antwort Erzeugung
def response(user_response):
    stop_words = get_stop_words('german')
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+ "Tut mir leid, ich verstehe dich nicht."
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

'''Ausgabe
(Um die Konsolenausgabe übersichtlicher zu gestalten wird die Bibliothek termcolor benutzt)'''
flag=True
clear = lambda: os.system('clear')
clear()
print(colored("CODY: ", 'green', attrs=['bold']) + colored("\tHallo, meine Name ist CODY. Ich weiß eine Menge über Chatbots. Frag' mich einfach!\n\tWenn du aufhören willst, tippe 'Bye'.", 'cyan'))
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='Danke dir' or user_response=='Danke' ):
            flag=False
            print(colored("CODY: ", 'green', attrs=['bold']) + colored( "Gerne..", 'cyan'))
        else:
            if(trivia(user_response)!=None):
                print(colored("CODY: ", 'green', attrs=['bold']) + colored(trivia(user_response), 'cyan'))
            else:
                print(colored("CODY: ", 'green', attrs=['bold']), end="")
                print(colored(response(user_response), 'cyan'))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print(colored("CODY: ", 'green', attrs=['bold']) + colored("Tschüss! Mach's gut.", 'cyan'))    
        
        