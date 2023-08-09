from django.shortcuts import render
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

# Create your views here.



def getpred(values):
    hd_scaler = pickle.load(open('hd_scaler.pkl', 'rb'))
    model = load_model('model.h5', compile=False)

    # values = [[70, 1, 4, 130, 322, 0, 2, 109, 0, 2.4, 2, 3]] # presence
    # values = [[67, 0, 3, 115, 564, 0, 2, 160, 1, 1.6, 2, 0]]  # Absence

    poss_results = ['Presence', 'Absence']
    encoder = LabelEncoder()
    encoder.fit_transform(poss_results)

    # encoding input
    values1 = hd_scaler.transform(values)
    pred = model.predict(values1)
    pred = [np.argmax(i) for i in pred]
    og_prediction = encoder.inverse_transform(pred)[0]
    return og_prediction


def predictor(request):
    if request.method == 'POST':
        print('inside loop')
        age = request.POST['age']
        gender = request.POST['gender']
        cpt = request.POST['cpt']
        bp = request.POST['bp']
        ekg = request.POST['ekg']
        chl = request.POST['chl']
        fbs = request.POST['fbs']
        ea = request.POST['ea']
        st_slope = request.POST['st_slope']
        hr = request.POST['hr']
        st = request.POST['st']
        fluro = request.POST['fluro']

        values = [[age, gender, cpt, bp, chl, fbs,
                ekg, hr, ea, st, st_slope, fluro]]

        pred = getpred(values)

        return render(request, 'index.html', { 'result' : pred })
    return render(request, 'index.html')
