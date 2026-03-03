# turbotrading

Budowa agenta AI do tradingu na rynku forex.

## Opis projektu

TurboTrading to agent AI przeznaczony do automatycznego handlu na rynku Forex. Projekt korzysta z technik uczenia maszynowego i analizy technicznej w celu identyfikowania okazji handlowych.

## Wymagania

- Python 3.10+
- Wirtualne środowisko (zalecane)

## Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/pitkaw5-design/turbotrading.git
cd turbotrading

# Utwórz i aktywuj wirtualne środowisko
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Zainstaluj zależności
pip install -r requirements.txt
```

## Użytkowanie

```bash
python -m turbotrading
```

## Konfiguracja

Skopiuj przykładowy plik konfiguracyjny i uzupełnij swoje dane API:

```bash
cp .env.example .env
```

## Licencja

Projekt licencjonowany na zasadach [Apache 2.0](LICENSE).

