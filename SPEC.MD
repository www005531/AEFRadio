# SPEC: Playlist Web (MP3 w tym folderze)

Minimalna aplikacja webowa w Pythonie (bez zewnetrznych zaleznosci) do zarzadzania i odtwarzania (streaming) playlist dla plikow `.mp3` znajdujacych sie **w tym samym folderze** co `playlist_web.py`.

## Cel

- Zarzadzanie playlistami (tworzenie/zmiana nazwy/usuwanie).
- Dodawanie/usuwanie mp3 do playlisty oraz zmiana kolejnosci.
- Eksport playlisty do `.m3u`.
- Odtwarzanie/streaming jako jeden endpoint audio: `/stream.mp3`.
- Tryb roboczy (draft): jednym URLem streamu da sie sterowac tym, co aktualnie gra, bez ponownego podpinania odtwarzacza.

## Wymagania i uruchomienie

- Python: **3.10+** (w kodzie sa typy `X | None`).
- Uruchomienie:
  - `python playlist_web.py`
  - przegladarka: `http://127.0.0.1:8001/`
- Parametry:
  - `--host` (domyslnie `127.0.0.1`)
  - `--port` (domyslnie `8001`)
- LAN (bezpieczenstwo: brak auth):
  - `python playlist_web.py --host 0.0.0.0 --port 8001`
- Skroty:
  - `playlist_web.bat` uruchamia `python playlist_web.py`

## Ograniczenia (twarde zalozenia)

- Obslugiwane sa **tylko** pliki `.mp3` z katalogu `BASE_DIR` (folder skryptu). Brak skanowania podfolderow.
- Aplikacja **nie** udostepnia innych plikow niz `.mp3` (poza HTML/JSON/M3U generowanymi przez serwer).
- Brak logowania/autoryzacji; bind na `0.0.0.0` udostepnia UI w sieci.

## Jezyk (PL/EN)

- UI ma przelacznik `Polski` / `English`.
- Parametr `lang=pl|en` (query) ustawia jezyk odpowiedzi HTML i zapisuje cookie `lang` dla kolejnych wejsc.
- Linki i formularze przenosza `lang`, zeby utrzymac wybrany jezyk po redirectach.

## Dane (SQLite)

- Plik bazy: `playlists.sqlite3` w `BASE_DIR` (obok `playlist_web.py`).
- Ustawienia:
  - `PRAGMA journal_mode = WAL` (lepsza rownoleglosc przy streamowaniu i klikaniu w UI)
  - `PRAGMA busy_timeout = 3000`
  - `PRAGMA foreign_keys = ON`

### Schemat

`playlists`

- `id` INTEGER PK autoincrement
- `name` TEXT NOT NULL UNIQUE
- `created_at` TEXT NOT NULL DEFAULT `datetime('now')`

`playlist_items`

- `playlist_id` INTEGER NOT NULL (FK -> `playlists.id`, `ON DELETE CASCADE`)
- `filename` TEXT NOT NULL
- `position` INTEGER NOT NULL (kolejnosc w playliscie)
- `added_at` TEXT NOT NULL DEFAULT `datetime('now')`
- PK: (`playlist_id`, `filename`)
- Index: `idx_playlist_items_pid_pos` na (`playlist_id`, `position`)

`playlist_stream_state`

- `playlist_id` INTEGER PK (FK -> `playlists.id`, `ON DELETE CASCADE`)
- `draft_mode` INTEGER NOT NULL DEFAULT 0 (0/1)
- `current_filename` TEXT NULL (ostatnio klikniete "Graj")
- `version` INTEGER NOT NULL DEFAULT 0 (inkrementowany przy zmianach, do natychmiastowego przelaczania streamu)
- `updated_at` TEXT NOT NULL DEFAULT `datetime('now')`

## UI (HTML)

### Strona glowna

- `GET /`
- Parametry:
  - `q` (opcjonalny): filtruje zarowno liste playlist (po nazwie), jak i liste mp3 (po nazwie pliku).
  - `lang` (opcjonalny): `pl` lub `en`
- Akcje:
  - `POST /playlist/create` tworzy playliste.
  - `POST /playlist/delete` usuwa playliste.
- Widoki:
  - tabela playlist + licznik utworow
  - tabela plikow mp3 w folderze + podglad w `<audio>` (endpoint `/mp3/...`)

### Strona playlisty

- `GET /playlist?id=<ID>`
- Parametry:
  - `id` (wymagany): ID playlisty
  - `q` (opcjonalny): filtruje zarowno liste "uwzglednione", jak i liste "do dodania"
  - `lang` (opcjonalny): `pl` lub `en`
- Widoki:
  - osadzony odtwarzacz streamu: `/stream.mp3?playlist_id=<ID>&loop=1`
  - linki `loop=1` i `loop=0`
  - przelacznik trybu roboczego (draft)
  - zmiana nazwy playlisty
  - tabela "Uwzglednione utwory" (pozycja, kolejnosc gora/dol, wykluczenie, lokalny podglad, przycisk `Graj`)
  - tabela "Dodaj utwor" (lista mp3 nie bedacych w playliscie)
    - dodawanie "bez przeladowania" przez `POST /api/playlist/add` (JS + JSON)

## Interfejs HTTP

### `GET /`

- HTML: lista playlist i lista mp3 w folderze.
- Query:
  - `q` (opcjonalny)
  - `lang` (opcjonalny): `pl` lub `en`

### `GET /playlist`

- Query:
  - `id` (int)
  - `q` (string, opcjonalnie)
  - `lang` (opcjonalny): `pl` lub `en`
  - `msg` / `err` (string, opcjonalnie; do komunikatow po redirectach)
- HTML: szczegoly playlisty.

### `GET /export.m3u`

- Query:
  - `playlist_id` (int)
- Odpowiedz:
  - `200 audio/x-mpegurl; charset=utf-8`
  - `Content-Disposition: attachment; filename="<sanitized>.m3u"`
  - format:
    - pierwsza linia `#EXTM3U`
    - dalej: 1 nazwa pliku na linie (sciezki wzgledne, same nazwy plikow)

### `GET /mp3/<filename>`

- Serwuje mp3 z `BASE_DIR`.
- Wspiera `Range: bytes=<start>-<end>`:
  - `206 Partial Content` + `Content-Range` gdy poprawny zakres
  - `Accept-Ranges: bytes`
- Walidacja:
  - tylko basename (brak `..`, brak `/` i `\\`)
  - wymagane rozszerzenie `.mp3`
  - plik musi istniec w `BASE_DIR`

### `GET /stream.mp3`

- Query:
  - `playlist_id` (int)
  - `loop` (0/1, domyslnie 1)
- `Content-Type: audio/mpeg`

#### `loop=1` (zalecane)

Jedno dlugotrwale polaczenie (chunked), ktore moze reagowac na zmiany w trakcie:

- Normalnie: odtwarza utwory z playlisty po `position`, a po koncu zapetla.
- Gdy wlaczony tryb roboczy (draft):
  - stream gra **tylko** `current_filename` (zapetlony)
  - jesli `current_filename` nie jest jeszcze ustawiony, serwer utrzymuje polaczenie i czeka
  - zmiana utworu nastepuje natychmiast po aktualizacji `version`

Pacing: serwer ogranicza predkosc wysylania bajtow do wartosci zblizonej do real-time (na podstawie wykrytego bitrate; fallback 160 kbps), z malym buforem burst. Uzywane sa male chunki (szczegolnie w draft), zeby przelaczanie bylo szybkie.

#### `loop=0`

- Gdy draft jest **wlaczony**: zwraca **pojedynczy** aktualnie wybrany utwor (`current_filename`) jako odpowiedz o znanej dlugosci (`Content-Length`).
- Gdy draft jest **wylaczony**: zwraca konkatenacje wszystkich plikow z playlisty (po kolei) jako jedna odpowiedz o znanej dlugosci (`Content-Length`).

### `POST /api/playlist/add`

- `Content-Type: application/x-www-form-urlencoded`
- Body:
  - `playlist_id` (int)
  - `filename` (string)
  - `q` (string, opcjonalnie; tylko do powrotu po redirectach w wersji nie-API)
  - `lang` (opcjonalny): `pl` lub `en`
- Odpowiedz JSON:
  - `200`: `{ "ok": true, "filename": "...", "position": <int> }`
  - `400`: `{ "ok": false, "error": "..." }`

### `POST /playlist/create`

- Body: `name`
- Body (opcjonalnie): `lang` (`pl` lub `en`)
- Redirect `303` do `/` z `msg` albo `err`.
- Walidacja: nazwa wymagana, trim, max 120, unikalna.

### `POST /playlist/delete`

- Body: `id` (+ opcjonalnie `lang`)
- Redirect `303` do `/` z `msg`.

### `POST /playlist/rename`

- Body: `id`, `name` (+ opcjonalnie `lang`)
- Redirect `303` do `/playlist?id=<id>` z `msg` albo `err`.
- Walidacja jak przy create.

### `POST /playlist/add`

- Body: `playlist_id`, `filename` (+ opcjonalnie `lang`)
- Dodaje mp3 na koniec (position = max+1).
- Redirect `303` do `/playlist`.

### `POST /playlist/remove`

- Body: `playlist_id`, `filename` (+ opcjonalnie `lang`)
- Usuwa utwor z playlisty i przepakuje `position` na 1..N.
- Redirect `303` do `/playlist`.

### `POST /playlist/move`

- Body: `playlist_id`, `filename`, `dir` = `up` lub `down` (+ opcjonalnie `lang`)
- Zamienia kolejnosc z sasiadem i aktualizuje `position`.
- Redirect `303` do `/playlist` (msg lub err).

### `POST /playlist/draft`

- Body: `playlist_id`, `enabled` = `0` lub `1` (+ opcjonalnie `lang`)
- Ustawia `draft_mode` oraz inkrementuje `version`.
- Redirect `303` do `/playlist` (msg lub err).

### `POST /playlist/play`

- Body: `playlist_id`, `filename` (+ opcjonalnie `lang`)
- Efekt:
  - ustawia `current_filename`
  - wlacza draft (jesli jeszcze nie wlaczony)
  - inkrementuje `version` (zeby stream `loop=1` przelaczyl utwor)
- Redirect `303` do `/playlist`.

## Walidacje i bledy (konwencje)

- MP3:
  - tylko pliki `.mp3` z `BASE_DIR`
  - blokada path traversal/absolutnych sciezek (tylko "bezpieczna nazwa pliku")
  - bledy walidacji zwracane jako `400` (dla `/mp3/...` i `/api/...`) albo jako `err` w query po redirectach dla widokow HTML
- Playlisty:
  - create/rename: wymagana nazwa, max 120, unikalnosc
- Statusy:
  - `404` gdy playlista nie istnieje lub (w przypadku streamu) brak wymaganych danych (np. draft wlaczony bez wybranego utworu przy `loop=0`)
