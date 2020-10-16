from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import sklearn

from pandas import Series
import pickle
#this is not important

filename = 'finalmodeldata.sav'
file2 = 'tfid.sav'
pac = pickle.load(open(filename, 'rb'))
tfidf_vectorizer = pickle.load(open(file2, 'rb'))
# very important
mainwin = Tk()
mainwin.state('zoomed')
mainwin.title("Fake News Detector")
#this is imp
'''
canv = Canvas(mainwin, width=500, height=500, bg='white')
canv.grid(row=0, column=0)

img = PhotoImage(file="1.png")
canv.create_image(20,20, anchor=NW, image=img)
'''
#this is not important

def clicked():
    new = st.get(1.0, END)
    new_series = Series(new)
    new2 = tfidf_vectorizer.transform(new_series)
    ypred2 = pac.predict(new2)
    messagebox.showinfo('Report', ypred2)


Label(mainwin, text="FAKE NEWS DETECTOR", font=(
    "Arial", 15), pady=10).grid(row=0, column=1)
st = ScrolledText(mainwin, wrap="word", width=135,
                  height=30, font=("Arial", 12))
st.grid(row=1, column=1, pady=10, padx=30)
st.insert(INSERT, """Enter News here""")

Button(mainwin, text="Predict", command=clicked,
       width=32).grid(row=4, column=1, sticky="EW")

mainwin.mainloop()
