# Ustawienia
$DropboxUrl = "https://www.dropbox.com/scl/fi/xu27w5dj4gc9z8bti2bkl/models.zip?rlkey=aaly9yiogosogweznn5xhokx8&dl=1"  

# Pobierz plik ZIP z Dropbox
Invoke-WebRequest -Uri $DropboxUrl -OutFile "models.zip"

# Rozpakuj plik ZIP
Expand-Archive -Path "models.zip" -DestinationPath "models\"

Write-Host "Pobieranie i rozpakowywanie zakończone."


