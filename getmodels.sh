#!/bin/bash

# Ustawienia
DROPBOX_URL="https://www.dropbox.com/scl/fi/xu27w5dj4gc9z8bti2bkl/models.zip?rlkey=aaly9yiogosogweznn5xhokx8&dl=1"  

echo "Pobieranie modeli..."

# Pobierz plik z Dropbox
curl -L -o "models.zip" "$DROPBOX_URL" 

echo "Pobieranie zakończone."

# Rozpakuj plik
unzip -o models.zip -d models/

echo "Rozpakowywanie zakończone."

# Usuń plik zip
rm models.zip

echo "Niepotrzebne pliki usunięte."