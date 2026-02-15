# AEFRadio: Playlist Web

Simple, dependency-free Python web app to manage playlists for `.mp3` files located **in the same folder** as `playlist_web.py`.

- App: `playlist_web.py`
- Spec: `SPEC.MD`

## English

### Requirements

- Python 3.10+

### Run

1. `python playlist_web.py`
2. Open `http://127.0.0.1:8001/`

Alternative: `playlist_web.bat`

LAN:

`python playlist_web.py --host 0.0.0.0 --port 8001`

### Data / Features

- Database: `playlists.sqlite3` (SQLite, local file in the folder)
- Export playlist: playlist page link `Export: .m3u`
- Stream a playlist as one audio endpoint: `/stream.mp3?playlist_id=<ID>&loop=1`
- Draft mode:
  - enable `Draft mode` on the playlist page
  - click `Play` next to a track
  - then `/stream.mp3` plays only the last clicked track (useful for quick switching)
- Tip: use `loop=1` so your player can stay attached to one URL and react to changes without reconnecting.

### Language

- Use the UI toggle in the header: `Polski` / `English`
- Or set via query string: `?lang=pl` / `?lang=en`

## Polski

### Wymagania

- Python 3.10+

### Uruchomienie

1. `python playlist_web.py`
2. Wejdz w przegladarce na `http://127.0.0.1:8001/`

Alternatywnie: `playlist_web.bat`

LAN:

`python playlist_web.py --host 0.0.0.0 --port 8001`

### Dane / Funkcje

- Baza danych: `playlists.sqlite3` (SQLite, lokalnie w folderze)
- Export playlisty: na stronie playlisty link `Export: .m3u`
- Streaming playlisty jako jeden strumien: `/stream.mp3?playlist_id=<ID>&loop=1`
- Tryb roboczy (draft):
  - na stronie playlisty wlacz `Tryb roboczy`
  - klikaj `Graj` przy utworach
  - wtedy `/stream.mp3` gra tylko ostatnio klikniety utwor (przydatne do szybkiego przelaczania)
- Wskazowka: uzywaj `loop=1`, bo wtedy odtwarzacz moze zostac podpiety do jednego URL i reaguje na zmiany bez ponownego startu odtwarzania.

### Jezyk

- Przelacznik w naglowku: `Polski` / `English`
- Albo parametr w URL: `?lang=pl` / `?lang=en`

