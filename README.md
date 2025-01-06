# LungAI - Instrukcja Instalacji i Uruchamiania

## O projekcie

Zapalenie płuc jest poważnym stanem chorobowym, który atakuje układ
oddechowy, prowadząc do zapalenia pęcherzyków płucnych, często wywołanego
przez bakterie, wirusy lub grzyby. Choroba ta stanowi istotne zagrożenie
zdrowotne, szczególnie w regionach o ograniczonym dostępie do opieki
medycznej, a jej wczesne wykrycie jest kluczowe w zapobieganiu powikłaniom i
zgonom. Tradycyjne metody diagnostyczne, takie jak badanie rentgenowskie
płuc, są często stosowane w celu identyfikacji zapalenia płuc, jednak ich
interpretacja może być subiektywna i zależna od doświadczenia radiologa. W
odpowiedzi na te wyzwania, celem niniejszego projektu inżynierskiego jest
stworzenie aplikacji, która przy pomocy algorytmów uczenia maszynowego
automatycznie analizuje zdjęcia rentgenowskie płuc, klasyfikując je jako
przedstawiające zapalenie płuc lub zdrowe płuca.

## Przygotowanie środowiska

Aby rozpocząć pracę z aplikacją **LungAI**, wykonaj poniższe kroki:

### 1. Sklonuj repozytorium aplikacji

```bash
git clone https://github.com/zuzanna56/LungAI.git
cd LungAI
```

### 2. Utwórz wirtualne środowisko

Aplikacja jest kompatybilna z **Python 3.11.6**. Utwórz wirtualne środowisko:

```bash
python -m venv env
source env/bin/activate   # Linux/MacOS
# lub
env\Scripts\activate     # Windows
```

### 3. Zainstaluj wymagane biblioteki

Zainstaluj wszystkie zależności z pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Instalacja bibliotek do generowania PDF

Aby umożliwić generowanie raportów PDF, zainstaluj dodatkowe biblioteki:

```bash
pip install pdfkit
pip install wtforms
```

### Instalacja `wkhtmltopdf`

Dla poprawnego działania `pdfkit`, wymagana jest instalacja narzędzia `wkhtmltopdf`.

#### Linux:

```bash
sudo apt install wkhtmltopdf
```

#### MacOS:

```bash
brew install --cask wkhtmltopdf
```

#### Windows:

1. Pobierz instalator `wkhtmltopdf` ze strony: [https://wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html).
2. Zainstaluj narzędzie, postępując zgodnie z instrukcjami na ekranie.
3. Zanotuj ścieżkę instalacji, np. `C:\Program Files\wkhtmltopdf`.
4. Dodaj ścieżkę `bin` z katalogu instalacyjnego do zmiennej środowiskowej `PATH`:
   - Otwórz **Panel Sterowania** → **System** → **Zaawansowane ustawienia systemu**.
   - W zakładce **Zaawansowane** kliknij **Zmienne środowiskowe**.
   - W sekcji **Zmienne systemowe** znajdź zmienną **Path** i kliknij **Edytuj**.
   - Dodaj nowy wpis: `C:\Program Files\wkhtmltopdf`.
   - Zapisz zmiany i zamknij wszystkie okna dialogowe.
5. Aby upewnić się, że `wkhtmltopdf` działa poprawnie, otwórz terminal i wpisz:

```bash
wkhtmltopdf --version
```

Jeśli wszystko zostało zainstalowane poprawnie, powinieneś zobaczyć wersję zainstalowanego oprogramowania.

---

## Uruchamianie aplikacji

Aby uruchomić aplikację, wykonaj następujące kroki:

1. Upewnij się, że wirtualne środowisko jest aktywne. Jeśli nie jest, aktywuj je:

```bash
source env/bin/activate   # Linux/MacOS
# lub
env\Scripts\activate     # Windows
```

2. Przejdź do katalogu `app`, gdzie znajduje się plik `app.py`:

```bash
cd app
```

3. Uruchom aplikację za pomocą następującej komendy:

```bash
flask run
```

4. Po uruchomieniu aplikacja będzie dostępna pod adresem: [http://127.0.0.1:5000](http://127.0.0.1:5000).

5. Aby zatrzymać działanie aplikacji, naciśnij `Ctrl+C` w terminalu.

---

## Dodatkowe informacje

- Upewnij się, że wszystkie wymagane biblioteki zostały zainstalowane poprawnie. W przeciwnym razie aplikacja może się nie uruchomić.
- W przypadku problemów sprawdź logi wyświetlane w terminalu – mogą one dostarczyć informacji o przyczynie błędu.
