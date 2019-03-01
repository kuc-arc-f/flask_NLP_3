# -*- coding: utf-8 -*-
# predict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from janome.tokenizer import Tokenizer
#
class NlpPredict:
    def __init__(self):
        self.params = {}
        self.words=[]
        self.answers =[]
        self.get_answers()
    #
    def get_token(self, text):
        t = Tokenizer()
        tokens = t.tokenize(text)
        word = ""
        for token in tokens:
            part_of_speech = token.part_of_speech.split(",")[0]
            if part_of_speech == "名詞":
                word +=token.surface + " "
            if part_of_speech == "動詞":
                word +=token.base_form+ " "
            if part_of_speech == "形容詞":
                word +=token.base_form+ " "
            if part_of_speech == "形容動詞":
                word +=token.base_form+ " "
        return word
    #
    def get_answers(self):
        ans1="利用人数は、通常プランは１０名までです。"
        ans2="契約は、１年、１カ月単位の契約が可能です"
        ans3="オープンソースです"
        ans4="オンライン決済は、可能です。"
        ans5="製品価格は、初期費用は無料です"
        ans6="雨の日でも、使えます"
        ans7="２４時間、利用可能です。 "
        ans8="寒い日も使えます。"
        self.answers =[]
        self.answers.append(ans1 )
        self.answers.append(ans2 )
        self.answers.append(ans3 )
        self.answers.append(ans4 )
        self.answers.append(ans5 )
        self.answers.append(ans6 )
        self.answers.append(ans7 )
        self.answers.append(ans8 )

    #
    def get_data(self):
        words1="利用人数は何人ですか？"
        words2="契約期間は、ありますか？"
        words3="オープンソースですか？"
        words4="オンライン決済は、可能ですか?"
        words5="製品価格、値段はいくらですか？"
        words6="雨の日は、使えますか？"
        words7="２４時間対応ですか？"
        words8="寒い日は、つかえますか？"
        #
        self.words =[]
        self.words.append(words1 )
        self.words.append(words2 )
        self.words.append(words3 )
        self.words.append(words4 )
        self.words.append(words5 )
        self.words.append(words6 )
        self.words.append(words7 )
        self.words.append(words8 )

        #print(words )
        tokens=[]
        for item in self.words:
            token=self.get_token(item)
            tokens.append(token)
        #
        #print(tokens )
        docs = np.array(tokens)
        return docs
    #
    def train(self, docs ):
        self.vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
        self.vecs = self.vectorizer.fit_transform(docs )
        return ""
    #
    def get_vectorize(self):
        return self.vectorizer
    #
    def predict(self, str):
        #str="利用人数は？"
#        str="契約期間"
        instr = self.get_token(str ).strip()
        #print("instr=", instr )
        x= self.vectorizer.transform( [  instr ])
        #print( "x=",x)
        #Cosine類似度（cosine_similarity）の算出
        num_sim=cosine_similarity(x , self.vecs)
        #print(num_sim )
        index = np.argmax( num_sim )
        #
        #print("word=", self.words[index])
        #print()
        #ret= self.words[index]
        ret= self.answers[ index ]
        return ret
