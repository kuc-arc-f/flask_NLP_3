# coding: utf-8
# 日本語

from flask import Flask
from flask import jsonify

class VectBase:
    #
    def __init__(self):
        from flaskr.include.nlp_predict import  NlpPredict
        self.pred=NlpPredict()
        #ans=self.pred.answers
        #print("ans-len=", len(ans))
        tokens=self.pred.get_data()
        #print(tokens )
        ret= self.pred.train(tokens )
        self.vectorize= self.pred.get_vectorize()
        print("#end-load-vectorize")
    #
    def predict(self, text ):
        text=self.pred.predict(text )
        #print(text )
        return text
#
app = Flask(__name__)
#app.config.from_object('flaskr.config')
app.config['JSON_AS_ASCII'] = False
#test()
vect=VectBase()
#text=vect.predict()
#print(text )

import flaskr.views