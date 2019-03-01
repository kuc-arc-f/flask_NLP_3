# 日本語

# coding: utf-8
#
from flask import request, redirect, url_for, render_template, flash
from flaskr import app, vect

from flask import jsonify
import numpy as np
#import json
#
@app.route('/')
def show_entries():
    return redirect(url_for('test3_show'))
    #return redirect(url_for('predict_show'))
#
@app.route('/test', methods=['GET', 'POST'])
def test():
    print("test")
    ret=vect.predict("")
    #print(ret )
    dic = {"text" : ret }
    return jsonify(dic)
#
@app.route('/test3', methods=['GET', 'POST'])
def test3():
    print("#test3")
    ret="sorry, nothing response."
    if(len(request.form ) > 0):
        text=request.form['intext']
        print("text=",text ,"len=", len(text) )
        if(len(text) >0):
            ret=vect.predict(text )
        print  (ret)
    return ret
#
@app.route('/test3_show', methods=['GET', 'POST'])
def test3_show():
    return render_template('chat.html' )
#
@app.route('/predict_show', methods=['GET', 'POST'])
def predict_show():
    return render_template('predict.html' )
#
