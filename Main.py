import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import easygui
import preprocess as pre
import Logisticregression as LR
import RF as RF
import DT as dt
import adaboost as ada
import GB as gb
import LGBM as lgbm
import MLP as mlp
import KNN as knn
import votingclassifier as vc
import time 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
        global window
        def clear():
            print("Clear1")
            txt.delete(0, 'end')    
            
  



        window = tk.Tk()
        window.title("DIAGNOSIS OF  POLYCYSTIC OVARY SYNDROME USING MACHINE LEARNING ALGORITHMS")
        
 
        window.geometry('1580x960')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="DIAGNOSIS OF  POLYCYSTIC OVARY SYNDROME USING ML" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=1)

        lbl = tk.Label(window, text="Dataset",width=10  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=1, y=100)
        
        txt = tk.Entry(window,width=30,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=200, y=110)
       

        


        def browse():
                path=filedialog.askopenfilename()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Datset")     

        
        def preproc():
                pre.process()
                print("preprocess")
                tm.showinfo("Input", "Preprocess Successfully Finished")
                
        def LRprocess():
                sym=txt.get()
                if sym != "":
                        LR.process(sym)
                        tm.showinfo("Input", "Logistic Regression Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")

        def RFprocess():
                sym=txt.get()
                if sym != "":
                        RF.process(sym)
                        tm.showinfo("Input", "Random Forest Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")

        def adaboostprocess():
                sym=txt.get()
                if sym != "":
                        ada.process(sym)
                        tm.showinfo("Input", "Adaboost Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")

        def DTprocess():
                sym=txt.get()
                if sym != "":
                        dt.process(sym)
                        tm.showinfo("Input", "Decision Tree Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")
        def GBprocess():
                sym=txt.get()
                if sym != "":
                        gb.process(sym)
                        tm.showinfo("Input", "Garient Boosting Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")
        def LGBMprocess():
                sym=txt.get()
                if sym != "":
                        lgbm.process(sym)
                        tm.showinfo("Input", "Light Garient Boosting Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")
        def VCprocess():
                sym=txt.get()
                if sym != "":
                        vc.process(sym)
                        tm.showinfo("Input", "MLP Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")
        def KNNprocess():
                sym=txt.get()
                if sym != "":
                        knn.process(sym)
                        tm.showinfo("Input", "KNN Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")

        def Predictprocess():
                #sym=txt2.get()
                Age =int(easygui.enterbox("Enter Age "))
                weight=float(easygui.enterbox("Weight (Kg)"))
                height=int(easygui.enterbox("Height(Cm)"))
                BMI=float(easygui.enterbox("BMI"))
                bgroup=easygui.enterbox("Blood Group(O+ ve)")
                if bgroup=="A+ ve":
                        bgroup= 11
                if bgroup=="A- ve":
                        bgroup= 12
                if bgroup=="B+ ve":
                        bgroup= 13
                if bgroup=="B- ve":
                        bgroup= 14
                if bgroup=="O+ ve":
                        bgroup= 15
                if bgroup=="O- ve":
                        bgroup= 16
                if bgroup=="AB+ ve":
                        bgroup= 17
                if bgroup=="AB- ve":
                        bgroup= 18
                paulerate=int(easygui.enterbox("Pulse rate(bpm) "))
                rr=int(easygui.enterbox("RR (breaths/min)"))
                hb=float(easygui.enterbox("Hb(g/dl)"))		
                ct=easygui.enterbox("Cycle(Regular/Irregular)")
                if ct=="Regular":
                        ct=2
                else:
                        ct=4
                cylen=int(easygui.enterbox("Cycle length(days)"))
                marry=int(easygui.enterbox("Marraige Status (Yrs)"))
                prag=easygui.enterbox("Pregnant(Yes/No)")
                if prag=="Yes":
                        prag=1
                else:
                        prag=0
                noa=int(easygui.enterbox("No. of aborptions"))
                IHCG=float(easygui.enterbox("I   beta-HCG(mIU/mL)"))
                IIHCG=float(easygui.enterbox("II   beta-HCG(mIU/mL)"))
                fsh=float(easygui.enterbox("FSH(mIU/mL)"))
                lh=float(easygui.enterbox("LH(mIU/mL)"))
                fshdividebylh=float(easygui.enterbox("FSH/LH"))
                hip=float(easygui.enterbox("Hip(inch)"))
                waist=float(easygui.enterbox("Waist(inch)"))
                whr=float(easygui.enterbox("Waist:Hip Ratio"))
                tsh=float(easygui.enterbox("TSH (mIU/L)"))
                amh=float(easygui.enterbox("AMH(ng/mL)"))
                prl=float(easygui.enterbox("PRL(ng/mL)"))
                v3=float(easygui.enterbox("Vit D3 (ng/mL)"))
                prg=float(easygui.enterbox("PRG(ng/mL)"))
                rbs=float(easygui.enterbox("RBS(mg/dl)"))
                
                wg=easygui.enterbox("Weight gain(Yes/No)")
                if wg=="Yes":
                        wg=1
                else:
                        wg=0
                hg=easygui.enterbox("hair growth(Yes/No)")
                if hg=="Yes":
                        hg=1
                else:
                        hg=0
                sd=easygui.enterbox("Skin darkening(Yes/No)")
                if sd=="Yes":
                        sd=1
                else:
                        sd=0
                hl=easygui.enterbox("Hair loss(Yes/No)")
                if hl=="Yes":
                        hl=1
                else:
                        hl=0
                pim=easygui.enterbox("Pimples (Yes/No)")
                if pim=="Yes":
                        pim=1
                else:
                        pim=0
                fd=easygui.enterbox("Fast food(Yes/No)")
                if fd=="Yes":
                        fd=1
                else:
                        fd=0
                re=easygui.enterbox("Reg.Exercise(Yes/No)")
                if re=="Yes":
                        re=1
                else:
                        re=0
                bpsys=float(easygui.enterbox("BP _Systolic (mmHg)"))
                bpdia=float(easygui.enterbox("BP _Diastolic (mmHg)"))
                fnol=float(easygui.enterbox("Follicle No. (L)"))
                fnor=float(easygui.enterbox("Follicle No. (R))"))
                afsl=float(easygui.enterbox("Avg. F size (L) (mm)"))
                afsr=float(easygui.enterbox("Avg. F size (R) (mm)"))
                endom=float(easygui.enterbox("Endometrium (mm)"))
                data=pd.read_csv("data.csv")    
                X=data.drop(["PCOS (Y/N)","Sl. No","Patient File No."],axis = 1) #droping out index from features too
                y=data["PCOS (Y/N)"]
                #Splitting the data into test and training sets
                X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
                #Fitting the RandomForestClassifier to the training set
                rfc = RandomForestClassifier()
                rfc.fit(X_train, y_train)
                x_text=[float(Age),float(weight),float(height),float(BMI),float(bgroup),float(paulerate),float(rr),float(hb),float(ct),float(cylen),float(marry),float(prag),float(noa),float(IHCG),float(IIHCG),float(fsh),float(lh),float(fshdividebylh),float(hip),float(waist),float(whr),float(tsh),float(amh),float(prl),float(v3),float(prg),float(rbs),float(wg),float(hg),float(sd),float(hl),float(pim),float(fd),float(re),float(bpsys),float(bpdia),float(fnol),float(fnor),float(afsl),float(afsr),float(endom)]
                print("x_text",x_text)
                inputdata=x_text
                x_text=np.array(x_text)
                x_text=x_text.reshape(1, -1)
                res=rfc.predict(x_text)
                msg=""
                if res[0]==0:
                        msg="No disease"
                else:
                        msg="You have infected by POCOS"
                tm.showinfo("Output", "Prediction : " +str(msg))



                

        browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=600, y=110)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=900, y=110)
         
        proc = tk.Button(window, text="Preprocess", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        proc.place(x=200, y=250)
        

        LRbutton = tk.Button(window, text="LogisticRegression", command=LRprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        LRbutton.place(x=200, y=350)


        RFbutton = tk.Button(window, text="RandomForest", command=RFprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        RFbutton.place(x=200, y=450)

        SVMbutton = tk.Button(window, text="AdaBoost", command=adaboostprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        SVMbutton.place(x=200, y=550)

        SVM1button = tk.Button(window, text="Decision Tree", command=DTprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        SVM1button.place(x=500, y=250)

        PRbutton1 = tk.Button(window, text="GB", command=GBprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        PRbutton1.place(x=500, y=350)

        RFbutton1 = tk.Button(window, text="LGBM", command=LGBMprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        RFbutton1.place(x=500, y=450)

        

        SVM1button2 = tk.Button(window, text="KNN", command=KNNprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        SVM1button2.place(x=500, y=550)
        SVM1button1 = tk.Button(window, text="Voting Classsifier", command=VCprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        SVM1button1.place(x=800, y=250)
        SVM1button3 = tk.Button(window, text="Predict", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        SVM1button3.place(x=800, y=350)


        

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=800, y=450)

        window.mainloop()
Home()

