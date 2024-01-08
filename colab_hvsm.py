from google.colab import drive
drive.mount("/content/drive", force_remount=True)
import os
from google.colab import output
import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
import random
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
import sys


imienazwisko="adam pospieszny"

# Nazwa naszego arkusza do którego beda kierowac sie wszystkie zebrane wyniki
arkusz = '11sl6PCw_9n_WPeu1wKH5nSjmRbmelGyiHj8IYShOZdM'


# Sciezka do zdjec, ktore chcemy wywolywac
sciezka = '/content/drive/MyDrive/Faces/'
HvsM = {}
randomowe = []

#sciezka do plikiu data klasyfikujacego zdjecia
csv_path = '/content/drive/MyDrive/Faces/data.csv'
df = pd.read_csv(csv_path)

#Funckja obliczajaca procent prawidlowo wytypowanych emocji w tej sesji
def prawidlowosesja():
    p = 0
    for i in HvsM:
      if HvsM[i][0] == HvsM[i][1]:
        p = 1 + p
    if p==0:
      return 0
    else:
      return ((p/len(HvsM))*100)

#Funkcja obliczajaca procent prawidlowych odpowiedzi na podstawie google sheets
def prawidlowosheets():
  gc = gspread.authorize(creds)
  worksheet = gc.open_by_key(arkusz).sheet1
  rows = worksheet.get_all_values()
  Path = [row[0] for row in rows[1:]]
  GT = [row[1] for row in rows[1:]]
  Rev = [row[2] for row in rows[1:]]
  var = 0
  total = 0
  for p in range(len(Path)-1, 0, -1):
      sciezka = Path[p]
      split_parts = sciezka.split('/')
      klasa = split_parts[0]
      if klasa != GT[p]:
          Path.pop(p)
          GT.pop(p)
  for il in range(len(Path)-1, 0, -1):
      if GT[il] == Rev[il]:
          var = 1 + var
      total = 1 + total
  return (var/total * 100)

#Funkcja majaca za zadanie usunac ostatnia komurke w google sheets
#jesli nie jest prawidlowo wypelniona
#Funkcja uruchamiana przez klawisz button7
def delLast(b):
    global h
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(arkusz).sheet1
    rows = worksheet.get_all_values()
    column_values = [row for row in rows[1:]]
    last = column_values[h-2]
    if  last[1] == '':
      worksheet.update_cell(h,1, '')
    raise ("")

#Funkcja generujaca liste zawierajaca wszystkie wczesniejsze ocenione zdjecia
def ex_oceny():
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(arkusz).sheet1
    rows = worksheet.get_all_values()
    column_values = [row[0] for row in rows[1:]]
    return column_values

#Funkcja odpowiadajaca za losowanie nowego obrazka z puli oraz
#upewnianie sie ze nie byl on juz wczesniej uzyty
def ran():
    while True:
        i = random.randrange(0, (df.shape[0]))
        klasa, nazwa = wywolanie(i)
        if nazwa not in ex_oceny():
            break
    randomowe.append(i)
    return i

#Funkcja generujaca numer nowego wiersza do zapisu w google sheets
def nowywiersz():
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(arkusz).sheet1
    rows = worksheet.get_all_values()
    nowy_wiersz = len(rows) + 1
    return nowy_wiersz

#Funkcja zapisujaca nazwe aktualnie ocenianego obrazka by go "zarezerwowac"
def rezerwacja_komurki(nazwa):
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(arkusz).sheet1
    wiersz = nowywiersz()
    zasieg = "A" + str(wiersz) + ":A" + str(wiersz)
    cell_list = worksheet.range(zasieg)
    for cell in cell_list:
      cell.value = nazwa
    worksheet.update_cells(cell_list)
    return wiersz

#Funkcja wypełniajaca zarezerwowany rzad wartosciami
def wypelnienie_komurki(nazwa, gt, rev, wiersz, varnazwisko="None"):
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(arkusz).sheet1
    rows = worksheet.get_all_values()
    last = rows.pop()
    zasieg = "A" + str(wiersz) + ":D" + str(wiersz)
    cell_list = worksheet.range(zasieg)
    g = 0
    for cell in cell_list:
      if g == 0:
        cell.value = nazwa
      if g == 1:
        cell.value = gt
      if g == 2:
        cell.value = rev
      if g == 3:
        cell.value = varnazwisko
      g = 1 + g

    worksheet.update_cells(cell_list)

#Funkcja odpowiedzialna za wyznaczenie Klasy i sciezki do zdjecia
def wywolanie(i=0):
    global df
    p = df.loc[i, 'path']
    split_parts = p.split('/')
    klasa = split_parts[0]
    nazwa = split_parts[1]
    return klasa, p

#Funkcja odswiezajaca i wywolujaca zdjecie
def load(i=0):
    klasa, nazwa = wywolanie(i)
    full_path = os.path.join(sciezka, nazwa)
    return full_path

#Funkcja generujaca plik csv z odpowiedzi z bierzacej sesji
def gen_csv(b):
    sciezka = []
    for i,j in HvsM.items():
      sciezka.append({"path":i,"GT":j[0],"Rev":j[1]})
    df2 = pd.DataFrame(sciezka)
    # sciezka pliku z inicjalami
    df2.to_csv("RevJS.csv")

def happy(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Happy", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Happy", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Happy")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data



def sad(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Sad", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Sad", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Sad")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data

def angry(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Angry", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Angry", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Angry")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data


def ahegao(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Ahegao", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Ahegao", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Ahegao")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data


def neutral(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Neutral", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Neutral", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Neutral")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data


def surprise(b):
    global i
    global h
    klasa, nazwa = wywolanie(i)
    if nowywiersz() == 2:
      h = 2
      rezerwacja_komurki(nazwa)
      wypelnienie_komurki(nazwa, klasa, "Surprise", h)
    else:
      wypelnienie_komurki(nazwa, klasa, "Surprise", nowywiersz(),imienazwisko)
    HvsM[nazwa] = (klasa,"Surprise")
    i = ran()
    klasa, nazwa = wywolanie(i)
    #h = rezerwacja_komurki(nazwa)
    jpg_path = load(i)
    with open(jpg_path, "rb") as image_file:
        image_data = image_file.read()
    ii.value = image_data


button = widgets.Button(description="HAPPY")
button.on_click(happy)

button1 = widgets.Button(description="SAD")
button1.on_click(sad)

button2 = widgets.Button(description="ANGRY")
button2.on_click(angry)

button3 = widgets.Button(description="AHEGAO")
button3.on_click(ahegao)

button4 = widgets.Button(description="NEUTRAL")
button4.on_click(neutral)

button5 = widgets.Button(description="SURPRISE")
button5.on_click(surprise)

button6 = widgets.Button(description="Generowanie CSV")
button6.on_click(gen_csv)

button7 = widgets.Button(description="Koniec Oceniania")
button7.on_click(delLast)

if nowywiersz==2:
  i=0
i = ran()
jpg_path = load(i)

with open(jpg_path, "rb") as image_file:
    image_data = image_file.read()

center_css = """
<style>
    .center {
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
"""



display(HTML(center_css))

ii = widgets.Image(value=image_data, format='jpg', width=300, height=300)
display(widgets.HBox([ii], layout=widgets.Layout(justify_content='center')))

display(widgets.HBox([button, button1, button2, button3, button4, button5], layout=widgets.Layout(justify_content='center')))
display(widgets.HBox([button6,button7], layout=widgets.Layout(justify_content='center')))
