from flask import Flask, render_template, jsonify, make_response, request, send_from_directory
import json
import requests
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# =============================================================
@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/hasil', methods = ['POST', 'GET'])
def hasil():
    try:
        Name = request.form['name'].upper()
        Age = int(request.form['age'])
        Total_Bilirubin = float(request.form['totalBilirubin'])
        Direct_Bilirubin = float(request.form['directBilirubin'])
        Alkaline_Phosphotase = int(request.form['alkalinePhosphotase'])
        Alamine_Aminotransferase = int(request.form['alamineAminotransferase'])
        Aspartate_Aminotransferase = int(request.form['aspartateAminotransferase'])
        Total_Protiens = float(request.form['totalProtiens'])
        Albumin = float(request.form['albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['albuminandGlobulinRatio'])
        
        print(Direct_Bilirubin)
        print(Alkaline_Phosphotase)
        print(type(Alkaline_Phosphotase))

        df = pd.read_csv('liver_new.csv')
        model = joblib.load('modelML')
        
        if Name=="" or Age=="" or Total_Bilirubin=="" or Direct_Bilirubin=="" or Alkaline_Phosphotase=="" or Alamine_Aminotransferase=="" or Aspartate_Aminotransferase=="" or Total_Protiens=="" or Albumin=="" or Albumin_and_Globulin_Ratio=="":
            return render_template('error.html')
        else:
            # prediksi SVM
            prediksi = model.predict([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
            
            if prediksi[0]==0:
                hasil = 'Liver'
            else:
                hasil = 'Normal'

            # Grafik pasien dengan nilai rata2 orang normal
            age1 = df[df['Dataset']==1]['Age'].mean()
            tb1= df[df['Dataset']==1]['Total_Bilirubin'].mean()
            db1 = df[df['Dataset']==1]['Direct_Bilirubin'].mean()
            ap1 = df[df['Dataset']==1]['Alkaline_Phosphotase'].mean()
            aa1 = df[df['Dataset']==1]['Alamine_Aminotransferase'].mean()
            aam1 = df[df['Dataset']==1]['Aspartate_Aminotransferase'].mean()
            tp1 = df[df['Dataset']==1]['Total_Protiens'].mean()
            a1 = df[df['Dataset']==1]['Albumin'].mean()
            agr1 = df[df['Dataset']==1]['Albumin_and_Globulin_Ratio'].mean()

            x = ['rata2', 'me']
            age2 = [age1, Age]
            tb2 = [tb1, Total_Bilirubin]
            db2 = [db1, Direct_Bilirubin]
            ap2 = [ap1, Alkaline_Phosphotase]
            aa2 = [aa1, Alamine_Aminotransferase]
            aam2 = [aam1, Aspartate_Aminotransferase]
            tp2 = [tp1, Total_Protien]
            a2 = [a1, Albumin]
            agr2 = [agr1, Albumin_and_Globulin_Ratio]

            plt.figure(figsize=(12,6))
            plt.subplot(241)
            plt.bar(x, age2, color=['blue', 'green'])
            plt.title('Umur')

            plt.subplot(242)
            plt.bar(x, tb2, color=['blue', 'green'])
            plt.title('Total Bilirubin')

            plt.subplot(243)
            plt.bar(x, db2, color=['blue', 'green'])
            plt.title('Direct Bilirubin')

            plt.subplot(244)
            plt.bar(x, ap2, color=['blue', 'green'])
            plt.title('Alkaline Phosphotase')

            plt.subplot(245)
            plt.bar(x, aa2, color=['blue', 'green'])
            plt.title('Alamine Aminotransferase')

            plt.subplot(246)
            plt.bar(x, aam2, color=['blue', 'green'])
            plt.title('Aspartate Aminotransferase')

            plt.subplot(247)
            plt.bar(x, tp2, color=['blue', 'green'])
            plt.title('Total Protien')

            plt.subplot(248)
            plt.bar(x, a2, color=['blue', 'green'])
            plt.title('Albumin')

            plt.subplot(251)
            plt.bar(x, agr2, color=['blue', 'green'])
            plt.title('Albumin and Globulin Ratio')

            i = 0
            while True:
                i += 1
                newname = '%s%s.png'%('filename', str(i))
                if os.path.exists('./storage/'+ newname):
                    continue
                plt.savefig('./storage/'+ newname)
                break

            grafik = 'http://localhost:5000/storage/'+ newname

            profil = {
                'name' : Name,
                'hasil' : hasil,
                'grafik': grafik
            }

            return render_template(
                'hasil.html',
                profil = profil
            )

    except:
        return render_template('error.html')        


@app.route('/storage/<namafile>')
def storage(namafile):
    return send_from_directory('./storage',namafile)


# not found display
@app.errorhandler(404)
def tidakfound(error):                                                 
    return make_response('<h1>NOT FOUND (404)</h1>')


if __name__ == '__main__':
    app.run(debug = True) 
