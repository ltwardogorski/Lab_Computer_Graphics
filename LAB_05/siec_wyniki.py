import plaidml.keras
plaidml.keras.install_backend()

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import load_model
import siec_model

historical_data = [11617, # 26 kwietnia 
                   11902, 
                   12218, 
                   12640, 
                   12877, 
                   13105, 
                   13375, # 3 maja
                   ]


for i in range(len(historical_data)):
    historical_data[i] /= siec_model.MAX_COVID_CASES
    

network = load_model('siec.h5')
file = open("output.txt", "w")

print("Przewidywania:")
file.writelines("Przewidywania:")
for i in range(15):
    prediction = network.predict(np.asarray(historical_data[i:i + siec_model.INPUT_COUNT]).reshape(1, siec_model.INPUT_COUNT))
    prediction = prediction[0][0]
    historical_data.append(prediction)
    print(str(3 + i + 1) + ' maja. Przypadki: ' + str(int(prediction * siec_model.MAX_COVID_CASES)))
    file.writelines('\n' + str(3 + i + 1) + ' maja. Przypadki: ' + str(int(prediction * siec_model.MAX_COVID_CASES)))

file.close()

