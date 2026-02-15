# Playlist Web (MP3 w tym folderze)

Minimalna aplikacja webowa w Pythonie do zarzadzania playlistami dla plikow `.mp3` znajdujacych sie **w tym samym folderze** co skrypt.

## Uruchomienie

1. `python playlist_web.py`
2. Wejdz w przegladarce na `http://127.0.0.1:8001/`

Alternatywnie: `playlist_web.bat`

## Dane

- Baza danych: `playlists.sqlite3` (SQLite, lokalnie w folderze)
- Export playlisty: na stronie playlisty link `Export: .m3u`
- Streaming playlisty jako jeden strumien: `/stream.mp3?playlist_id=<ID>&loop=1`
- Tryb "wersji roboczej": na stronie playlisty wlacz `Tryb roboczy`, a potem klikaj `Graj` przy utworach. Wtedy `/stream.mp3` gra tylko ostatnio klikniety utwor.
- Wskazowka: uzywaj `loop=1`, bo wtedy odtwarzacz moze zostac podpiety do jednego URL i reaguje na zmiany bez ponownego startu odtwarzania.

## Uwaga (sieciowo)

Jesli chcesz odpalic na LAN:

`python playlist_web.py --host 0.0.0.0 --port 8001`
