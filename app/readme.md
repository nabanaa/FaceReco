# Make a Face

### Aplikacja Make-A-Face - gra w "robienie twarzy"

Gra polegająca wykonywaniu ekspresji które podaje nam gra u góry ekranu. Jeśli wykonamy ekspresję która odpowiada tej o którą prosiła nas aplikacja, zostanie naliczony nam punkt. Jako gracz próbujemy skumulować jak najwięcej punktów w trakcie 20 sekund.

## Instalacja

Aby zainstalować wymagane biblioteki, należy użyć polecenia:

`pip install -r requirements.txt`

### Wymagania

- Python 3.11+
- Pip

## User Interface

W grze mamy następujące przyciski:

    Play - Rozpoczyna rozgrywkę.
    Pause - Wstrzymuje rozgrywkę.
    New Game - Rozpoczyna nową grę, umożliwia zmianę gracza.
    Highscores - Wyświetla najlepsze wyniki graczy w obecnej sesji.

Po naciśnięciu przycisku Play, gra zostaje uruchomiona, a detekcja twarzy rozpoczyna się w czasie rzeczywistym. Możesz zatrzymać rozpoznawanie, naciskając przycisk Pause. Aby rozpocząć nową grę, naciśnij przycisk New Game.

Po zakończeniu każdej rundy, wynik zostanie zapisany, a najlepsze wyniki będą dostępne w sekcji Highscores. Aby zakończyć grę, użyj przycisku zamykania okna.

W aplikacji znajduje się również możliwość wyłączenia kategorii "Ahegao" za pomocą opcji No-Ahegao. Wprowadź imię gracza w polu Player przed rozpoczęciem gry.

![A very surprised man](/images/surprise_maf.png "A very surprised man")
