#Sieć neuronowa

#Bibioteki do obliczen tensorowych

#import tensorflow as tf
#from tensorflow import keras

import plaidml.keras
plaidml.keras.install_backend()

#Bibioteka do obsługi sieci neuronowych
import keras

#Załadowania bazy uczącej
import imageio
import numpy as np

import os

from keras.models import load_model

# returns a compiled model
# identical to the previous one
genderModel = load_model('siec.h5')
genderModel.summary() # Display summary

ImgWidth = 100
ImgHeight = 100

BazaImg = np.empty((50,ImgHeight,ImgWidth,3))


FileName = ".\\baza_testowa\\K\\1.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[0,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\2.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[1,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\3.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[2,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\4.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[3,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\5.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[4,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\6.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[5,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\7.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[6,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\8.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[7,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\9.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[8,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\10.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[9,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\11.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[10,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\12.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[11,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\13.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[12,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\14.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[13,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\15.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[14,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\16.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[15,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\17.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[16,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\18.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[17,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\19.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[18,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\20.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[19,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\21.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[20,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\22.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[21,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\23.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[22,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\24.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[23,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\25.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[24,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]


#------------------------------------------------------------

FileName = ".\\baza_testowa\\M\\1.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[25,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\2.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[26,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\3.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[27,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\4.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[28,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\5.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[29,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\6.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[30,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\7.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[31,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\8.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[32,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\9.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[33,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\10.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[34,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\11.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[35,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\12.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[36,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\13.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[37,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\14.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[38,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\15.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[39,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\16.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[40,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\17.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[41,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\18.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[42,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\19.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[43,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\20.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[44,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\21.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[45,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\22.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[46,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\23.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[47,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\24.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[48,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\25.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[49,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]


gender = genderModel.predict(BazaImg) # 0 - m, 1 - k
#print(gender)

real_woman = 0
real_man = 0

complex_man = 0
complex_woman = 0

for i in range (0,25):
    if (gender[i]>0.5) :
        real_woman=real_woman+1
    else :
        complex_woman=complex_woman+1

for i in range (25,50):
    if (gender[i]<0.5) :
        real_man=real_man+1
    else :
        complex_man=complex_man+1

file = open("output.txt", "w")

file.writelines("+-----------------------+----------------------+\n") 
file.writelines("|                       | Faktyczna plec osoby |\n")
file.writelines("|                       | na zdjeciu           |\n")
file.writelines("+                       +----------+-----------+\n")
file.writelines("|                       |  Kobieta | Mezczyzna |\n")
file.writelines("+-----------+-----------+----------+-----------+\n")
file.writelines("| Odpowiedz |  Kobieta  |    {}    |     {}     |\n".format(real_woman,complex_man))
file.writelines("| sieci     +-----------+----------+-----------+\n")
file.writelines("|           | Mezczyzna |    {}     |     {}    |\n".format(complex_woman,real_man))
file.writelines("+-----------------------+----------------------+\n") 

file.close()


'''
+-----------------------+----------------------+
|                       | Faktyczna plec osoby |
|                       | na zdjeciu           |
+                       +----------+-----------+
|                       |  Kobieta | Mezczyzna |
+-----------+-----------+----------+-----------+
| Odpowiedz |  Kobieta  |    16    |     1     |
| sieci     +-----------+----------+-----------+
|           | Mezczyzna |     9    |    24     |
+-----------+-----------+----------+-----------+
'''
