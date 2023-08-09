import numpy as np
from sklearn.preprocessing import LabelEncoder

import pickle
hd_scaler = pickle.load(open('hd_scaler.pkl', 'rb'))
# model = pickle.load(open('heart_disease_model.pkl', 'rb'))

from keras.models import load_model
model = load_model('model.h5', compile=False)


# values = [[70, 1, 4, 130, 322, 0, 2, 109, 0, 2.4, 2, 3]] # presence
values = [[67, 0, 3, 115, 564, 0, 2, 160, 1, 1.6, 2, 0]]  # Absence


poss_results = ['Presence', 'Absence']
encoder = LabelEncoder()
encoder.fit_transform(poss_results)


# encoding input
values1 = hd_scaler.transform(values)

pred = model.predict(values1)

pred = [np.argmax(i) for i in pred]
og_prediction = encoder.inverse_transform(pred)[0]
print(og_prediction, flush=True)
