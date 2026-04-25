import tkinter as tk
import json
from catboost import CatBoostClassifier
from signs_coder import coder_no_svd
from numpy import int64
import pandas as pd

with open('heroes_list.json', 'r', encoding='utf-8') as f:
    heroes = json.load(f)
with open('teams_list.json', 'r', encoding='utf-8') as f:
    teams = json.load(f)
X = [None] * 6
radiant_heroes = [None] * 5
dire_heroes = [None] * 5
X[4] = radiant_heroes
X[5] = dire_heroes
model = CatBoostClassifier()
model.load_model('model_catboost_cpu_no_svd.json')

last_focused_entry = None
class vector_Entry(tk.Entry):
    def __init__(self, vector, cell, **kwargs):
        super().__init__(**kwargs)
        self.vector = vector
        self.cell = cell

    def fill_cell(self, value):
        if isinstance(self.cell, tuple):
            self.vector[self.cell[0]][self.cell[1]] = value
        else:
            self.vector[self.cell] = value

def input_hero(event):
    global last_focused_entry
    obj = event.widget
    last_focused_entry = obj
    
    value = obj.get().lower()
    
    if value == '':
        listbox.place_forget()
    else:
        data = [hero for hero in heroes if value in hero.lower()]
        if data:
            listbox.delete(0, tk.END)
            for hero in data:
                listbox.insert(tk.END, hero)
            
            listbox.place(x=obj.winfo_x(), y=obj.winfo_y() + obj.winfo_height(), width=obj.winfo_width())
            listbox.lift()
        else:
            listbox.place_forget()

def input_team(event):
    global last_focused_entry
    obj = event.widget
    last_focused_entry = obj
    
    value = obj.get().lower()
    
    if value == '':
        listbox.place_forget()
    else:
        data = [team for team in teams if value in team.lower()]
        if data:
            listbox.delete(0, tk.END)
            for team in data:
                listbox.insert(tk.END, team)
            
            listbox.place(x=obj.winfo_x(), y=obj.winfo_y() + obj.winfo_height(), width=obj.winfo_width())
            listbox.lift()
        else:
            listbox.place_forget()

def on_select(event):
    if last_focused_entry:
        selection = listbox.curselection()
        if selection:
            selected_item = listbox.get(selection[0])
            
            last_focused_entry.delete(0, tk.END)
            last_focused_entry.insert(0, selected_item)
            last_focused_entry.fill_cell(selected_item)
            
            listbox.place_forget()
            last_focused_entry.focus_set() # Возвращаем фокус в поле ввода

def input_streak(event):
    obj = event.widget
    value = obj.get().lower()
    try:
        value = int64(value)
    except ValueError:
        return
    obj.fill_cell(value)

def call_predict(event=None):
    if not (None in X) and not (None in X[4]) and not (None in X[5]):
        X_m = pd.DataFrame({
            'radiant_team': [X[0]],
            'dire_team': [X[1]], 
            'radiant_streak': [X[2]], 
            'dire_streak': [X[3]], 
            'radiant_heroes': [X[4]], 
            'dire_heroes': [X[5]]
        })
        X_m = coder_no_svd(X_m)
        Y = model.predict_proba(X_m)[0][0]
        if Y > 0.5:
            predict.config(text=f'Radiant with {(Y * 100):.2f}%', fg='green')
        else:
            predict.config(text=f'Dire with {((1 - Y) * 100):.2f}%', fg='red')
    else:
        predict.config(text=f'Err: Empty fields', fg='yellow')

root = tk.Tk(className='oracle_prediction')
root.title('oracle prediction')
root.geometry('1210x180')

canvas = tk.Canvas(root, width=1210, height=180)
canvas.create_line(0, 89, 1039, 89, width=2, fill='grey')
canvas.create_line(1039, 0, 1039, 179, width=2, fill='grey')
canvas.pack()
# Creating 10 fields for heroes.
for i, j in enumerate([0, 2, 3, 4, 1]):
    entry = vector_Entry(vector = X, cell = (4, j), master = root, font=('Arial', 11))
    entry.place(x=309 + 145 * i, y=39, width=140, height=40)
    entry.bind('<KeyRelease>', input_hero)
    tk.Label(root, text=i + 1, font=('Arial', 9)).place(x=309 + 145 * i, y=20)
for i, j in enumerate([0, 2, 3, 4, 1]):
    entry = vector_Entry(vector = X, cell = (5, j), master = root, font=('Arial', 11))
    entry.place(x=309 + 145 * i, y=99, width=140, height=40)
    entry.bind('<KeyRelease>', input_hero)
    tk.Label(root, text=i + 1, font=('Arial', 9)).place(x=309 + 145 * i, y=139)
# Creating 2 fields for teams.
for i in range(2):
    entry = vector_Entry(vector = X, cell = i, master = root, font=('Arial', 11))
    entry.place(x=19, y=39 + 60 * i, width=140, height=40)
    entry.bind('<KeyRelease>', input_team)
    tk.Label(root, text='Team', font=('Arial', 9)).place(x=19, y=38 + 101 * i + 18 * (i - 1))
# Creating 2 fields for streaks.
for i in range(2):
    entry = vector_Entry(vector = X, cell = i + 2, master = root, font=('Arial', 11))
    entry.place(x=164, y=39 + 60 * i, width=140, height=40)
    entry.bind('<KeyRelease>', input_streak)
    tk.Label(root, text='Series streak', font=('Arial', 9)).place(x=164, y=38 + 101 * i + 18 * (i - 1))
tk.Label(root, text='Radiant', font=('Arial', 11, 'bold')).place(x=9, y=4, height=15)
tk.Label(root, text='Dire', font=('Arial', 11, 'bold')).place(x=9, y=164, height=15)
predict = tk.Label(root, text='', font=('Arial', 12), bd=4, relief='ridge')
predict.place(x=1049, y=39, width=140, height=40)
call_button = tk.Button(root, text='Predict', bg='green', font=('Arial', 11, 'bold'), bd=4, relief='raised', command=call_predict)
call_button.place(x=1049, y=99, width=140, height=40)
listbox = tk.Listbox(root, font=('Arial', 12))
listbox.bind('<<ListboxSelect>>', on_select)

root.mainloop()