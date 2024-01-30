# Camera Capture

### Narzędzie do wykrywania i katalogowania twarzy

Narzędzie do wykrywania i katalogowania twarzy w obrazach. Wykorzystuje bibliotekę OpenCV do wykrywania twarzy, a następnie zapisuje je na dysku użytkownika.

## Instalacja

Aby zainstalować wymagane biblioteki, należy użyć polecenia:

`pip install -r requirements.txt`

### Wymagania

- Python 3.11+
- Pip

## Użycie

Aby uruchomić program, należy użyć polecenia:

`python camera_capture/CameraCapture.py`

Dla jak najlepszego skatalogowania twarzy, należy wybrać odpowiedni klawisz.

Kolejno:

- `h (ang. happy) - szczęśliwa miny`
- `d (ang. sad) - smutna mina`
- `s (ang. suprised) - zaskoczona mina`
- `n (ang. neutral) - neutralna mina`
- `a (ang. angry) - zła mina`
- `o - ahegao`
- `q - zakończenie działania programu`

Po użyciu odpowiedniego klawisza zdjęcie jest zrobione natychmiastowo i zapisane w odpowiednim folderze. Po zakończeniu pracy należy wcisnąć klawisz q, aby zakończyć działanie programu.
Do stworzenia własnego zbioru danych treningowych, należy ręcznie wgrać zdjęcia na Dysk Google, a następnie skorzystać z notatnika **TrainYourOwnModel.ipynb**.
