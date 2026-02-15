#!/usr/bin/env python3
# Simple playlist manager for MP3 files in the current folder (no external deps).
#
# Run:
#   python playlist_web.py
# Open:
#   http://127.0.0.1:8001/
#
# Data:
#   playlists.sqlite3 (in this folder)

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "playlists.sqlite3"
FILE_CHUNK_SIZE = 256 * 1024
# Smaller chunks help the "draft" stream switch tracks quickly after clicking "Graj".
STREAM_CHUNK_SIZE = 16 * 1024
DRAFT_STREAM_CHUNK_SIZE = 4 * 1024

SUPPORTED_LANGS = ("pl", "en")
DEFAULT_LANG = "pl"

TRANSLATIONS: dict[str, dict[str, str]] = {
    "pl": {
        # Common
        "language_label": "Jezyk:",
        "lang_pl": "Polski",
        "lang_en": "English",
        "label_ok": "OK",
        "label_error": "BLAD",
        "http_not_found": "Nie znaleziono.",
        "label_missing": "BRAK",
        "pill_stream": "STREAM",
        # Home
        "home_page_title": "Playlisty",
        "home_heading": "Zarzadzanie playlistami (MP3 w folderze)",
        "home_subtitle": "Folder: {folder} | Baza: {db} | {time}",
        "home_search_placeholder": "Szukaj: playlista lub plik mp3",
        "playlist_search_placeholder": "Szukaj w mp3 / utworach playlisty",
        "btn_search": "Szukaj",
        "section_playlists": "Playlisty",
        "placeholder_new_playlist": "Nowa playlista",
        "btn_create": "Utworz",
        "btn_delete": "Usun",
        "confirm_delete_playlist": "Usunac playliste?",
        "table_name": "Nazwa",
        "table_tracks": "Utwory",
        "table_file": "Plik",
        "table_preview": "Podglad",
        "pill_track_count": "{count} utwor(ow)",
        "no_playlists": "Brak playlist.",
        "section_mp3": "Pliki MP3 (biezacy folder)",
        "no_mp3": "Brak mp3 (albo filtr nic nie znalazl).",
        # Playlist
        "not_found_title": "Nie znaleziono",
        "playlist_not_found": "Nie znaleziono playlisty",
        "nav_back": "Powrot",
        "nav_stream": "Stream",
        "nav_export": "Export",
        "nav_id": "ID",
        "playlist_title": "Playlista: {name}",
        "draft_pill_on": "Tryb roboczy: WL",
        "draft_pill_off": "Tryb roboczy: WYL",
        "section_draft": "Tryb roboczy (sterowanie strumieniem)",
        "draft_btn_enable": "Wlacz tryb roboczy",
        "draft_btn_disable": "Wylacz tryb roboczy",
        "draft_hint": "Gdy WL: <code>/stream.mp3</code> gra tylko ostatnio klikniety przycisk <b>{play_btn}</b> (w tabeli).",
        "draft_current": "Aktualnie na streamie (roboczy): <code>{current}</code>",
        "section_rename": "Zmiana nazwy",
        "btn_rename": "Zmien",
        "section_included": "Uwzglednione utwory",
        "section_add_quick": "Dodaj utwor (szybko, bez przeladowania)",
        "placeholder_quick_filter": "Filtruj mp3 do dodania...",
        "table_pos": "#",
        "table_order": "Kolejnosc",
        "table_playlist": "Playlista",
        "table_play": "Odtwarzaj",
        "btn_exclude": "Wyklucz",
        "btn_include": "Uwzglednij",
        "btn_play": "Graj",
        "title_up": "Gora",
        "title_down": "Dol",
        "no_included": "Brak uwzglednionych utworow.",
        "no_excluded": "Brak plikow do dodania.",
        # JS
        "js_added": "Dodano do playlisty.",
        "js_add_failed": "Nie udalo sie dodac.",
        "js_error_generic": "Blad.",
        # Errors
        "err_invalid_filename": "Nieprawidlowa nazwa pliku.",
        "err_only_mp3": "Dozwolone sa tylko pliki .mp3.",
        "err_file_not_found": "Plik nie istnieje w biezacym folderze.",
        "err_playlist_not_found": "Nie znaleziono playlisty.",
        "err_playlist_name_required": "Nazwa playlisty jest wymagana.",
        "err_playlist_name_too_long": "Nazwa playlisty jest za dluga.",
        "err_playlist_exists": "Playlista o takiej nazwie juz istnieje.",
        "err_track_already_in_playlist": "Ten plik jest juz w playliscie.",
        "err_invalid_direction": "Nieprawidlowy kierunek.",
        "err_track_not_in_playlist": "Utwor nie jest w playliscie.",
        "err_draft_disabled": "Tryb roboczy jest wylaczony.",
        "err_draft_no_selection": "Tryb roboczy: brak wybranego utworu. Kliknij '{play_btn}' przy mp3.",
        "err_draft_prefix": "Tryb roboczy: {err}",
        "err_playlist_no_valid_mp3": "Playlista nie ma zadnych poprawnych plikow mp3.",
        # Messages
        "msg_playlist_created": "Utworzono playliste.",
        "msg_playlist_deleted": "Usunieto playliste.",
        "msg_playlist_renamed": "Zmieniono nazwe.",
        "msg_track_added": "Dodano utwor.",
        "msg_playlist_changed": "Zmieniono playliste.",
        "msg_order_changed": "Zmieniono kolejnosc.",
        "msg_draft_enabled": "Wlaczono tryb roboczy. Kliknij '{play_btn}' przy utworze.",
        "msg_draft_disabled": "Wylaczono tryb roboczy.",
        "msg_stream_set": "Ustawiono na stream: {filename}",
    },
    "en": {
        # Common
        "language_label": "Language:",
        "lang_pl": "Polski",
        "lang_en": "English",
        "label_ok": "OK",
        "label_error": "ERROR",
        "http_not_found": "Not found.",
        "label_missing": "MISSING",
        "pill_stream": "STREAM",
        # Home
        "home_page_title": "Playlists",
        "home_heading": "Playlist management (MP3 in folder)",
        "home_subtitle": "Folder: {folder} | DB: {db} | {time}",
        "home_search_placeholder": "Search: playlist or MP3 file",
        "playlist_search_placeholder": "Search in MP3 / playlist tracks",
        "btn_search": "Search",
        "section_playlists": "Playlists",
        "placeholder_new_playlist": "New playlist",
        "btn_create": "Create",
        "btn_delete": "Delete",
        "confirm_delete_playlist": "Delete playlist?",
        "table_name": "Name",
        "table_tracks": "Tracks",
        "table_file": "File",
        "table_preview": "Preview",
        "pill_track_count": "{count} track(s)",
        "no_playlists": "No playlists.",
        "section_mp3": "MP3 files (current folder)",
        "no_mp3": "No MP3 files (or the filter matched nothing).",
        # Playlist
        "not_found_title": "Not found",
        "playlist_not_found": "Playlist not found",
        "nav_back": "Back",
        "nav_stream": "Stream",
        "nav_export": "Export",
        "nav_id": "ID",
        "playlist_title": "Playlist: {name}",
        "draft_pill_on": "Draft: ON",
        "draft_pill_off": "Draft: OFF",
        "section_draft": "Draft mode (stream control)",
        "draft_btn_enable": "Enable draft mode",
        "draft_btn_disable": "Disable draft mode",
        "draft_hint": "When ON: <code>/stream.mp3</code> plays only the last clicked <b>{play_btn}</b> button (in the table).",
        "draft_current": "Currently on stream (draft): <code>{current}</code>",
        "section_rename": "Rename",
        "btn_rename": "Rename",
        "section_included": "Included tracks",
        "section_add_quick": "Add track (quick, no reload)",
        "placeholder_quick_filter": "Filter MP3 files to add...",
        "table_pos": "#",
        "table_order": "Order",
        "table_playlist": "Playlist",
        "table_play": "Play",
        "btn_exclude": "Exclude",
        "btn_include": "Include",
        "btn_play": "Play",
        "title_up": "Up",
        "title_down": "Down",
        "no_included": "No included tracks.",
        "no_excluded": "No files to add.",
        # JS
        "js_added": "Added to playlist.",
        "js_add_failed": "Failed to add.",
        "js_error_generic": "Error.",
        # Errors
        "err_invalid_filename": "Invalid file name.",
        "err_only_mp3": "Only .mp3 files are allowed.",
        "err_file_not_found": "File does not exist in the current folder.",
        "err_playlist_not_found": "Playlist not found.",
        "err_playlist_name_required": "Playlist name is required.",
        "err_playlist_name_too_long": "Playlist name is too long.",
        "err_playlist_exists": "A playlist with this name already exists.",
        "err_track_already_in_playlist": "This file is already in the playlist.",
        "err_invalid_direction": "Invalid direction.",
        "err_track_not_in_playlist": "Track is not in the playlist.",
        "err_draft_disabled": "Draft mode is disabled.",
        "err_draft_no_selection": "Draft mode: no selected track. Click '{play_btn}' next to an MP3.",
        "err_draft_prefix": "Draft mode: {err}",
        "err_playlist_no_valid_mp3": "Playlist has no valid MP3 files.",
        # Messages
        "msg_playlist_created": "Playlist created.",
        "msg_playlist_deleted": "Playlist deleted.",
        "msg_playlist_renamed": "Playlist renamed.",
        "msg_track_added": "Track added.",
        "msg_playlist_changed": "Playlist updated.",
        "msg_order_changed": "Order updated.",
        "msg_draft_enabled": "Draft mode enabled. Click '{play_btn}' next to a track.",
        "msg_draft_disabled": "Draft mode disabled.",
        "msg_stream_set": "Stream set to: {filename}",
    },
}


def normalize_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def t(lang: str, key: str, **kwargs: Any) -> str:
    lang = normalize_lang(lang)
    template = TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS[DEFAULT_LANG].get(key) or key
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


BITRATES_KBPS: dict[tuple[str, int], list[int]] = {
    # (mpeg_version_group, layer) where mpeg_version_group is "1" for MPEG1 else "2" for MPEG2/2.5
    ("1", 1): [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    ("1", 2): [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    ("1", 3): [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    ("2", 1): [32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    ("2", 2): [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    ("2", 3): [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}

SAMPLE_RATES: dict[int, list[int]] = {
    3: [44100, 48000, 32000],  # MPEG1
    2: [22050, 24000, 16000],  # MPEG2
    0: [11025, 12000, 8000],  # MPEG2.5
}


def _synchsafe_to_int(b: bytes) -> int:
    if len(b) != 4:
        return 0
    return ((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14) | ((b[2] & 0x7F) << 7) | (b[3] & 0x7F)


def mp3_bitrate_kbps(path: Path) -> int | None:
    # Best-effort: read the first valid MP3 frame header and extract bitrate.
    try:
        with path.open("rb") as f:
            head = f.read(128 * 1024)
    except Exception:
        return None

    if len(head) < 4:
        return None

    start = 0
    if head.startswith(b"ID3") and len(head) >= 10:
        tag_size = _synchsafe_to_int(head[6:10])
        start = min(len(head), 10 + tag_size)

    for i in range(start, len(head) - 4):
        b0 = head[i]
        b1 = head[i + 1]
        if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
            continue

        version_id = (b1 >> 3) & 0x03
        layer_id = (b1 >> 1) & 0x03
        if version_id == 1 or layer_id == 0:
            continue

        b2 = head[i + 2]
        bitrate_idx = (b2 >> 4) & 0x0F
        sample_idx = (b2 >> 2) & 0x03
        padding = (b2 >> 1) & 0x01

        if bitrate_idx in (0, 15) or sample_idx == 3:
            continue

        layer = {3: 1, 2: 2, 1: 3}.get(layer_id)
        if not layer:
            continue

        ver_group = "1" if version_id == 3 else "2"
        table = BITRATES_KBPS.get((ver_group, layer))
        if not table:
            continue
        bitrate_kbps = table[bitrate_idx - 1]

        sr_table = SAMPLE_RATES.get(version_id)
        if not sr_table:
            continue
        sample_rate = sr_table[sample_idx]

        # Validate by checking the next frame sync when possible (reduces false positives).
        frame_len = 0
        if layer == 1:
            frame_len = int((12 * (bitrate_kbps * 1000) / sample_rate + padding) * 4)
        elif layer == 2:
            frame_len = int(144 * (bitrate_kbps * 1000) / sample_rate + padding)
        else:
            coef = 144 if version_id == 3 else 72
            frame_len = int(coef * (bitrate_kbps * 1000) / sample_rate + padding)

        if frame_len <= 4:
            continue
        j = i + frame_len
        if j + 1 < len(head):
            nb0 = head[j]
            nb1 = head[j + 1]
            if nb0 != 0xFF or (nb1 & 0xE0) != 0xE0:
                continue

        return bitrate_kbps

    return None


class RateLimiter:
    def __init__(self, bytes_per_second: float, burst_seconds: float = 0.5) -> None:
        self.bps = max(1.0, float(bytes_per_second))
        self.burst = max(0.0, float(burst_seconds))
        self.t0 = time.monotonic()
        self.sent = 0

    def on_send(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        self.sent += nbytes
        desired = (self.sent / self.bps) - self.burst
        delay = desired - (time.monotonic() - self.t0)
        if delay > 0:
            time.sleep(delay)


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 3000;")
    return conn


def db_init() -> None:
    with db_connect() as conn:
        # Improve read/write concurrency (important while streaming + clicking in UI).
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS playlists (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS playlist_items (
              playlist_id INTEGER NOT NULL,
              filename TEXT NOT NULL,
              position INTEGER NOT NULL,
              added_at TEXT NOT NULL DEFAULT (datetime('now')),
              PRIMARY KEY (playlist_id, filename),
              FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_playlist_items_pid_pos
              ON playlist_items(playlist_id, position);

            CREATE TABLE IF NOT EXISTS playlist_stream_state (
              playlist_id INTEGER PRIMARY KEY,
              draft_mode INTEGER NOT NULL DEFAULT 0,
              current_filename TEXT,
              version INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
            );
            """
        )


def list_mp3_files() -> list[str]:
    names: list[str] = []
    for p in BASE_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".mp3":
            names.append(p.name)
    names.sort(key=lambda s: s.lower())
    return names


def is_safe_filename(name: str) -> bool:
    if not name:
        return False
    # Block any path traversal or absolute paths.
    if name != Path(name).name:
        return False
    if "/" in name or "\\" in name:
        return False
    return True


def validate_mp3_name(name: str, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    if not is_safe_filename(name):
        return False, t(lang, "err_invalid_filename")
    if not name.lower().endswith(".mp3"):
        return False, t(lang, "err_only_mp3")
    path = BASE_DIR / name
    if not path.exists() or not path.is_file():
        return False, t(lang, "err_file_not_found")
    return True, ""


def html_page(lang: str, title: str, body: str) -> bytes:
    css = """
    :root { color-scheme: light; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
           max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #111; }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
    h1 { font-size: 20px; margin: 0 0 16px; }
    h2 { font-size: 16px; margin: 20px 0 10px; }
    .muted { color: #555; font-size: 12px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    input[type="text"], input[type="search"] { padding: 8px 10px; border: 1px solid #bbb; border-radius: 8px; min-width: 280px; }
    button { padding: 8px 10px; border: 1px solid #333; border-radius: 8px; background: #fff; cursor: pointer; }
    button.icon { padding: 6px 10px; min-width: 36px; }
    button[disabled] { opacity: 0.45; cursor: not-allowed; }
    button.danger { border-color: #8a1f1f; color: #8a1f1f; }
    a { color: #0b57d0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #e5e5e5; padding: 8px 6px; vertical-align: top; }
    th { text-align: left; color: #333; font-weight: 600; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { display: inline-block; font-size: 12px; padding: 2px 8px; border: 1px solid #ddd; border-radius: 999px; color: #333; }
    .missing { color: #8a1f1f; font-weight: 600; }
    .ok { color: #0d6a3a; font-weight: 600; }
    .msg { padding: 10px 12px; border: 1px solid #ddd; border-radius: 10px; background: #fafafa; }
    audio { width: 280px; height: 32px; }
    """

    doc = f"""<!doctype html>
<html lang="{html.escape(normalize_lang(lang))}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
    return doc.encode("utf-8", errors="strict")


def qp_first(qs: dict[str, list[str]], key: str, default: str = "") -> str:
    vals = qs.get(key)
    if not vals:
        return default
    return vals[0]


def as_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def redirect_location(path: str, qs: dict[str, str] | None = None) -> str:
    if not qs:
        return path
    parts: list[str] = []
    for k, v in qs.items():
        parts.append(f"{quote(k)}={quote(v)}")
    return path + "?" + "&".join(parts)


@dataclass(frozen=True)
class Playlist:
    id: int
    name: str
    track_count: int


@dataclass(frozen=True)
class StreamState:
    playlist_id: int
    draft_mode: bool
    current_filename: str | None
    version: int


def db_get_stream_state(pid: int) -> StreamState | None:
    with db_connect() as conn:
        exists = conn.execute("SELECT 1 FROM playlists WHERE id = ?;", (pid,)).fetchone()
        if not exists:
            return None
        conn.execute("INSERT OR IGNORE INTO playlist_stream_state(playlist_id) VALUES (?);", (pid,))
        row = conn.execute(
            "SELECT playlist_id, draft_mode, current_filename, version FROM playlist_stream_state WHERE playlist_id = ?;",
            (pid,),
        ).fetchone()
    if not row:
        return None
    return StreamState(
        playlist_id=int(row["playlist_id"]),
        draft_mode=bool(int(row["draft_mode"])),
        current_filename=str(row["current_filename"]) if row["current_filename"] is not None else None,
        version=int(row["version"]),
    )


def db_set_draft_mode(pid: int, enabled: bool, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    with db_connect() as conn:
        exists = conn.execute("SELECT 1 FROM playlists WHERE id = ?;", (pid,)).fetchone()
        if not exists:
            return False, t(lang, "err_playlist_not_found")
        conn.execute("INSERT OR IGNORE INTO playlist_stream_state(playlist_id) VALUES (?);", (pid,))
        conn.execute(
            """
            UPDATE playlist_stream_state
            SET draft_mode = ?,
                version = version + 1,
                updated_at = datetime('now')
            WHERE playlist_id = ?;
            """,
            (1 if enabled else 0, pid),
        )
    return True, ""


def db_set_stream_track(pid: int, filename: str, enable_draft: bool = True, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    ok, err = validate_mp3_name(filename, lang)
    if not ok:
        return False, err
    with db_connect() as conn:
        exists = conn.execute("SELECT 1 FROM playlists WHERE id = ?;", (pid,)).fetchone()
        if not exists:
            return False, t(lang, "err_playlist_not_found")
        conn.execute("INSERT OR IGNORE INTO playlist_stream_state(playlist_id) VALUES (?);", (pid,))
        conn.execute(
            """
            UPDATE playlist_stream_state
            SET current_filename = ?,
                draft_mode = CASE WHEN ? THEN 1 ELSE draft_mode END,
                version = version + 1,
                updated_at = datetime('now')
            WHERE playlist_id = ?;
            """,
            (filename, 1 if enable_draft else 0, pid),
        )
    return True, ""


def db_list_playlists(search: str) -> list[Playlist]:
    search = (search or "").strip()
    like = f"%{search}%"
    with db_connect() as conn:
        if search:
            rows = conn.execute(
                """
                SELECT p.id, p.name, COUNT(i.filename) AS track_count
                FROM playlists p
                LEFT JOIN playlist_items i ON i.playlist_id = p.id
                WHERE p.name LIKE ?
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE;
                """,
                (like,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT p.id, p.name, COUNT(i.filename) AS track_count
                FROM playlists p
                LEFT JOIN playlist_items i ON i.playlist_id = p.id
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE;
                """
            ).fetchall()
    return [Playlist(id=int(r["id"]), name=str(r["name"]), track_count=int(r["track_count"])) for r in rows]


def db_create_playlist(name: str, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, t(lang, "err_playlist_name_required")
    if len(name) > 120:
        return False, t(lang, "err_playlist_name_too_long")
    with db_connect() as conn:
        try:
            conn.execute("INSERT INTO playlists(name) VALUES (?);", (name,))
        except sqlite3.IntegrityError:
            return False, t(lang, "err_playlist_exists")
    return True, ""


def db_delete_playlist(pid: int) -> None:
    with db_connect() as conn:
        conn.execute("DELETE FROM playlists WHERE id = ?;", (pid,))


def db_rename_playlist(pid: int, name: str, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, t(lang, "err_playlist_name_required")
    if len(name) > 120:
        return False, t(lang, "err_playlist_name_too_long")
    with db_connect() as conn:
        try:
            cur = conn.execute("UPDATE playlists SET name = ? WHERE id = ?;", (name, pid))
            if cur.rowcount == 0:
                return False, t(lang, "err_playlist_not_found")
        except sqlite3.IntegrityError:
            return False, t(lang, "err_playlist_exists")
    return True, ""


def db_get_playlist(pid: int) -> tuple[int, str] | None:
    with db_connect() as conn:
        row = conn.execute("SELECT id, name FROM playlists WHERE id = ?;", (pid,)).fetchone()
    if not row:
        return None
    return int(row["id"]), str(row["name"])


def db_list_playlist_tracks(pid: int, search: str) -> list[tuple[str, int]]:
    search = (search or "").strip()
    like = f"%{search}%"
    with db_connect() as conn:
        if search:
            rows = conn.execute(
                """
                SELECT filename, position
                FROM playlist_items
                WHERE playlist_id = ? AND filename LIKE ?
                ORDER BY position ASC;
                """,
                (pid, like),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT filename, position
                FROM playlist_items
                WHERE playlist_id = ?
                ORDER BY position ASC;
                """,
                (pid,),
            ).fetchall()
    return [(str(r["filename"]), int(r["position"])) for r in rows]


def db_add_track(pid: int, filename: str, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    ok, err = validate_mp3_name(filename, lang)
    if not ok:
        return False, err
    with db_connect() as conn:
        exists = conn.execute("SELECT 1 FROM playlists WHERE id = ?;", (pid,)).fetchone()
        if not exists:
            return False, t(lang, "err_playlist_not_found")
        max_pos = conn.execute(
            "SELECT COALESCE(MAX(position), 0) AS m FROM playlist_items WHERE playlist_id = ?;",
            (pid,),
        ).fetchone()["m"]
        next_pos = int(max_pos) + 1
        try:
            conn.execute(
                "INSERT INTO playlist_items(playlist_id, filename, position) VALUES (?, ?, ?);",
                (pid, filename, next_pos),
            )
        except sqlite3.IntegrityError:
            return False, t(lang, "err_track_already_in_playlist")
    return True, ""


def db_remove_track(pid: int, filename: str) -> None:
    if not is_safe_filename(filename):
        return
    with db_connect() as conn:
        conn.execute(
            "DELETE FROM playlist_items WHERE playlist_id = ? AND filename = ?;",
            (pid, filename),
        )
        # Re-pack positions to keep ordering stable and clean.
        rows = conn.execute(
            "SELECT filename FROM playlist_items WHERE playlist_id = ? ORDER BY position ASC;",
            (pid,),
        ).fetchall()
        for idx, r in enumerate(rows, start=1):
            conn.execute(
                "UPDATE playlist_items SET position = ? WHERE playlist_id = ? AND filename = ?;",
                (idx, pid, str(r["filename"])),
            )


def db_move_track(pid: int, filename: str, direction: int, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    if not is_safe_filename(filename):
        return False, t(lang, "err_invalid_filename")
    if direction not in (-1, 1):
        return False, t(lang, "err_invalid_direction")
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT filename FROM playlist_items WHERE playlist_id = ? ORDER BY position ASC;",
            (pid,),
        ).fetchall()
        ordered = [str(r["filename"]) for r in rows]
        if filename not in ordered:
            return False, t(lang, "err_track_not_in_playlist")
        idx = ordered.index(filename)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(ordered):
            return True, ""
        ordered[idx], ordered[new_idx] = ordered[new_idx], ordered[idx]
        for pos, fn in enumerate(ordered, start=1):
            conn.execute(
                "UPDATE playlist_items SET position = ? WHERE playlist_id = ? AND filename = ?;",
                (pos, pid, fn),
            )
    return True, ""


def html_lang_switcher(lang: str, path: str, params: dict[str, str]) -> str:
    lang = normalize_lang(lang)
    base = dict(params)
    base.pop("lang", None)

    pl_url = redirect_location(path, {**base, "lang": "pl"})
    en_url = redirect_location(path, {**base, "lang": "en"})

    def pill(code: str, label: str, url: str) -> str:
        if lang == code:
            return f"<span class='pill'>{html.escape(label)}</span>"
        return f"<a class='pill' href='{html.escape(url, quote=True)}'>{html.escape(label)}</a>"

    return (
        f"<span class='muted'>{html.escape(t(lang, 'language_label'))}</span>"
        f"{pill('pl', t(lang, 'lang_pl'), pl_url)}"
        f"{pill('en', t(lang, 'lang_en'), en_url)}"
    )


def html_home(lang: str, search: str, msg: str, err: str) -> bytes:
    playlists = db_list_playlists(search)
    mp3 = list_mp3_files()
    s = (search or "").strip().lower()
    if s:
        mp3 = [n for n in mp3 if s in n.lower()]

    msg_html = ""
    if msg:
        msg_html = f'<div class="msg"><span class="ok">{html.escape(t(lang, "label_ok"))}</span> {html.escape(msg)}</div>'
    elif err:
        msg_html = f'<div class="msg"><span class="missing">{html.escape(t(lang, "label_error"))}</span> {html.escape(err)}</div>'

    rows = []
    safe_lang = normalize_lang(lang)
    confirm_js = t(lang, "confirm_delete_playlist").replace("\\", "\\\\").replace("'", "\\'")
    for p in playlists:
        href = redirect_location("/playlist", {"id": str(p.id), "lang": safe_lang})
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(href, quote=True)}">{html.escape(p.name)}</a></td>'
            f'<td><span class="pill">{html.escape(t(lang, "pill_track_count", count=p.track_count))}</span></td>'
            f"<td class='muted'><form method='post' action='/playlist/delete' onsubmit=\"return confirm('{confirm_js}');\">"
            f"<input type='hidden' name='id' value='{p.id}' />"
            f"<input type='hidden' name='lang' value='{html.escape(safe_lang)}' />"
            f"<button class='danger' type='submit'>{html.escape(t(lang, 'btn_delete'))}</button>"
            f"</form></td>"
            "</tr>"
        )
    playlists_table = (
        "<table>"
        f"<thead><tr><th>{html.escape(t(lang, 'table_name'))}</th><th>{html.escape(t(lang, 'table_tracks'))}</th><th></th></tr></thead>"
        "<tbody>"
        + ("".join(rows) if rows else f"<tr><td colspan='3' class='muted'>{html.escape(t(lang, 'no_playlists'))}</td></tr>")
        + "</tbody></table>"
    )

    mp3_rows = []
    for n in mp3:
        mp3_rows.append(
            "<tr>"
            f"<td><code>{html.escape(n)}</code></td>"
            f"<td><audio controls preload='none' src='/mp3/{quote(n)}'></audio></td>"
            "</tr>"
        )
    mp3_table = (
        "<table>"
        f"<thead><tr><th>{html.escape(t(lang, 'table_file'))}</th><th>{html.escape(t(lang, 'table_preview'))}</th></tr></thead>"
        "<tbody>"
        + ("".join(mp3_rows) if mp3_rows else f"<tr><td colspan='2' class='muted'>{html.escape(t(lang, 'no_mp3'))}</td></tr>")
        + "</tbody></table>"
    )

    lang_switcher = html_lang_switcher(safe_lang, "/", {"q": search})
    subtitle = t(
        lang,
        "home_subtitle",
        folder=html.escape(str(BASE_DIR)),
        db=html.escape(DB_PATH.name),
        time=html.escape(utc_now_iso()),
    )

    body = f"""
<header>
  <div>
    <h1>{html.escape(t(lang, "home_heading"))}</h1>
    <div class="muted">{subtitle}</div>
  </div>
  <div class="row">
    {lang_switcher}
    <form method="get" action="/" class="row">
      <input type="hidden" name="lang" value="{html.escape(safe_lang)}" />
      <input type="search" name="q" value="{html.escape(search)}" placeholder="{html.escape(t(lang, "home_search_placeholder"))}" />
      <button type="submit">{html.escape(t(lang, "btn_search"))}</button>
    </form>
  </div>
</header>
{msg_html}

<h2>{html.escape(t(lang, "section_playlists"))}</h2>
<form method="post" action="/playlist/create" class="row">
  <input type="hidden" name="lang" value="{html.escape(safe_lang)}" />
  <input type="text" name="name" placeholder="{html.escape(t(lang, "placeholder_new_playlist"))}" required />
  <button type="submit">{html.escape(t(lang, "btn_create"))}</button>
</form>
{playlists_table}

<h2>{html.escape(t(lang, "section_mp3"))}</h2>
{mp3_table}
"""
    return html_page(lang, t(lang, "home_page_title"), body)


def html_playlist(lang: str, pid: int, search: str, msg: str, err: str) -> bytes:
    pl = db_get_playlist(pid)
    if not pl:
        safe_lang = normalize_lang(lang)
        back_href = redirect_location("/", {"lang": safe_lang})
        body = (
            f"<h1>{html.escape(t(lang, 'playlist_not_found'))}</h1>"
            f"<p><a href='{html.escape(back_href, quote=True)}'>{html.escape(t(lang, 'nav_back'))}</a></p>"
        )
        return html_page(lang, t(lang, "not_found_title"), body)
    _, name = pl
    stream_state = db_get_stream_state(pid)
    draft_mode = bool(stream_state.draft_mode) if stream_state else False
    current_stream_fn = stream_state.current_filename if stream_state else None
    all_tracks = db_list_playlist_tracks(pid, "")
    pos_by_fn = {fn: pos for (fn, pos) in all_tracks}
    ordered_fns = [fn for (fn, _pos) in all_tracks]
    index_by_fn = {fn: idx for idx, fn in enumerate(ordered_fns)}

    folder_files = list_mp3_files()
    s = (search or "").strip().lower()
    included_files = [fn for fn in ordered_fns if (not s or (s in fn.lower()))]
    included_set = set(ordered_fns)
    excluded_files = [fn for fn in folder_files if (fn not in included_set and (not s or (s in fn.lower())))]

    safe_lang = normalize_lang(lang)
    ok_badge = html.escape(t(lang, "label_ok"))
    err_badge = html.escape(t(lang, "label_error"))
    missing_badge = html.escape(t(lang, "label_missing"))
    stream_badge = html.escape(t(lang, "pill_stream"))

    msg_html = ""
    if msg:
        msg_html = f'<div class="msg"><span class="ok">{ok_badge}</span> {html.escape(msg)}</div>'
    elif err:
        msg_html = f'<div class="msg"><span class="missing">{err_badge}</span> {html.escape(err)}</div>'

    if draft_mode:
        draft_label = f"<span class='pill'>{html.escape(t(lang, 'draft_pill_on'))}</span>"
        draft_btn_label = t(lang, "draft_btn_disable")
        draft_next = "0"
    else:
        draft_label = f"<span class='pill'>{html.escape(t(lang, 'draft_pill_off'))}</span>"
        draft_btn_label = t(lang, "draft_btn_enable")
        draft_next = "1"
    current_esc = html.escape(current_stream_fn) if current_stream_fn else "-"
    draft_current_line = t(lang, "draft_current", current=current_esc)

    included_rows = []
    for fn in included_files:
        pos = pos_by_fn.get(fn)
        exists = (BASE_DIR / fn).exists()
        status = f"<span class='ok'>{ok_badge}</span>" if exists else f"<span class='missing'>{missing_badge}</span>"
        stream_pill = f" <span class='pill'>{stream_badge}</span>" if (draft_mode and current_stream_fn == fn) else ""

        idx = index_by_fn.get(fn, 0)
        can_up = idx > 0
        can_down = idx < (len(ordered_fns) - 1)
        up_disabled = "" if can_up else " disabled"
        down_disabled = "" if can_down else " disabled"
        order_controls = (
            "<div class='row'>"
            f"<form method='post' action='/playlist/move'>"
            f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
            f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
            f"<input type=\"hidden\" name=\"dir\" value=\"up\" />"
            f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
            f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
            f"<button class='icon' data-move='up' type='submit'{up_disabled} title='{html.escape(t(lang, 'title_up'))}'>&uarr;</button>"
            f"</form>"
            f"<form method='post' action='/playlist/move'>"
            f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
            f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
            f"<input type=\"hidden\" name=\"dir\" value=\"down\" />"
            f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
            f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
            f"<button class='icon' data-move='down' type='submit'{down_disabled} title='{html.escape(t(lang, 'title_down'))}'>&darr;</button>"
            f"</form>"
            "</div>"
        )

        toggle = (
            f"<form method='post' action='/playlist/remove' class='row'>"
            f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
            f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
            f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
            f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
            f"<button type='submit'>{html.escape(t(lang, 'btn_exclude'))}</button>"
            f"</form>"
        )

        if exists:
            player = (
                "<div class='row'>"
                f"<form method='post' action='/playlist/play' class='row'>"
                f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
                f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
                f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
                f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
                f"<button type='submit'>{html.escape(t(lang, 'btn_play'))}</button>"
                f"</form>"
                f"<audio controls preload='none' src='/mp3/{quote(fn)}'></audio>"
                "</div>"
            )
        else:
            player = "<span class='muted'>-</span>"

        included_rows.append(
            "<tr data-fn='{fn_lc}' data-filename='{fn_raw}'>".format(fn_lc=html.escape(fn.lower()), fn_raw=html.escape(fn))
            + f"<td>{pos}</td>"
            + f"<td><code>{html.escape(fn)}</code>{stream_pill} {status}</td>"
            + f"<td>{order_controls}</td>"
            + f"<td>{toggle}</td>"
            + f"<td>{player}</td>"
            + "</tr>"
        )

    included_table = (
        "<table>"
        f"<thead><tr><th>{html.escape(t(lang, 'table_pos'))}</th><th>{html.escape(t(lang, 'table_file'))}</th><th>{html.escape(t(lang, 'table_order'))}</th><th>{html.escape(t(lang, 'table_playlist'))}</th><th>{html.escape(t(lang, 'table_play'))}</th></tr></thead>"
        "<tbody id='includedBody'>"
        + ("".join(included_rows) if included_rows else f"<tr><td colspan='5' class='muted'>{html.escape(t(lang, 'no_included'))}</td></tr>")
        + "</tbody></table>"
    )

    excluded_rows = []
    for fn in excluded_files:
        exists = (BASE_DIR / fn).exists()
        status = f"<span class='ok'>{ok_badge}</span>" if exists else f"<span class='missing'>{missing_badge}</span>"
        stream_pill = f" <span class='pill'>{stream_badge}</span>" if (draft_mode and current_stream_fn == fn) else ""

        toggle = (
            f"<form method='post' action='/playlist/add' class='row js-include'>"
            f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
            f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
            f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
            f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
            f"<button type='submit'>{html.escape(t(lang, 'btn_include'))}</button>"
            f"</form>"
        )

        if exists:
            player = (
                "<div class='row'>"
                f"<form method='post' action='/playlist/play' class='row'>"
                f"<input type=\"hidden\" name=\"playlist_id\" value=\"{pid}\" />"
                f"<input type=\"hidden\" name=\"filename\" value=\"{html.escape(fn)}\" />"
                f"<input type=\"hidden\" name=\"q\" value=\"{html.escape(search)}\" />"
                f"<input type=\"hidden\" name=\"lang\" value=\"{html.escape(safe_lang)}\" />"
                f"<button type='submit'>{html.escape(t(lang, 'btn_play'))}</button>"
                f"</form>"
                f"<audio controls preload='none' src='/mp3/{quote(fn)}'></audio>"
                "</div>"
            )
        else:
            player = "<span class='muted'>-</span>"

        excluded_rows.append(
            "<tr data-fn='{fn_lc}' data-filename='{fn_raw}'>".format(fn_lc=html.escape(fn.lower()), fn_raw=html.escape(fn))
            + f"<td><code>{html.escape(fn)}</code>{stream_pill} {status}</td>"
            + f"<td>{toggle}</td>"
            + f"<td>{player}</td>"
            + "</tr>"
        )

    excluded_table = (
        "<table>"
        f"<thead><tr><th>{html.escape(t(lang, 'table_file'))}</th><th></th><th>{html.escape(t(lang, 'table_play'))}</th></tr></thead>"
        "<tbody id='excludedBody'>"
        + ("".join(excluded_rows) if excluded_rows else f"<tr><td colspan='3' class='muted'>{html.escape(t(lang, 'no_excluded'))}</td></tr>")
        + "</tbody></table>"
    )

    # Simple client-side filtering + include without full page reload.
    js_i18n = {
        "ok": t(lang, "label_ok"),
        "err": t(lang, "label_error"),
        "add_failed": t(lang, "js_add_failed"),
        "added": t(lang, "js_added"),
        "generic_error": t(lang, "js_error_generic"),
        "up": t(lang, "title_up"),
        "down": t(lang, "title_down"),
        "exclude": t(lang, "btn_exclude"),
        "play": t(lang, "btn_play"),
    }
    js_L = json.dumps(js_i18n, ensure_ascii=False)

    js = f"""
<script>
(() => {{
  const filter = document.getElementById('quickFilter');
  const excludedBody = document.getElementById('excludedBody');
  const includedBody = document.getElementById('includedBody');
  const msg = document.getElementById('jsMsg');
  if (!excludedBody) return;

  const L = {js_L};

  const esc = (s) => String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('\"', '&quot;')
    .replaceAll(\"'\", '&#x27;');

  const applyFilter = () => {{
    if (!filter) return;
    const q = (filter.value || '').trim().toLowerCase();
    const rows = Array.from(excludedBody.querySelectorAll('tr[data-fn]'));
    let shown = 0;
    for (const r of rows) {{
      const fn = (r.getAttribute('data-fn') || '');
      const ok = (!q || fn.includes(q));
      r.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }}
    const cnt = document.getElementById('quickCount');
    if (cnt) cnt.textContent = `${{shown}} / ${{rows.length}}`;
  }};

  if (filter) {{
    filter.addEventListener('input', applyFilter);
    applyFilter();
  }}

  const showMsg = (text, isErr) => {{
    if (!msg) return;
    msg.style.display = 'block';
    msg.innerHTML = isErr ? `<span class="missing">${{esc(L.err)}}</span> ${{esc(text)}}` : `<span class="ok">${{esc(L.ok)}}</span> ${{esc(text)}}`;
    window.clearTimeout(msg._t);
    msg._t = window.setTimeout(() => {{ msg.style.display = 'none'; }}, 2500);
  }};

  document.addEventListener('submit', async (ev) => {{
    const form = ev.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!form.classList.contains('js-include')) return;
    ev.preventDefault();

    const fd = new FormData(form);
    const playlistId = fd.get('playlist_id');
    const filename = fd.get('filename');
    const langVal = String(fd.get('lang') || '');
    if (!playlistId || !filename) return;

    const btn = form.querySelector('button[type=\"submit\"]');
    if (btn) btn.disabled = true;

    try {{
      const res = await fetch('/api/playlist/add', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
        body: new URLSearchParams(fd),
      }});
      const data = await res.json();
      if (!data || !data.ok) {{
        throw new Error((data && data.error) ? data.error : L.add_failed);
      }}

      // Remove from excluded list.
      const tr = form.closest('tr');
      if (tr) tr.remove();
      applyFilter();

      // Append to included list (as last).
      if (includedBody) {{
        const pos = data.position || '?';
        const fn = String(filename);
        const fnUrl = encodeURIComponent(fn);
        const qVal = String(fd.get('q') || '');

        // Enable down-arrow on previous last included row (if any).
        const last = includedBody.querySelector('tr:last-child');
        if (last) {{
          const downBtn = last.querySelector('button[data-move=\"down\"]');
          if (downBtn) downBtn.disabled = false;
        }}

        const canUp = (pos !== 1 && pos !== '1');
        const upDisabled = canUp ? '' : ' disabled';
        const rowHtml = `
          <tr data-fn="${{esc(fn.toLowerCase())}}" data-filename="${{esc(fn)}}">
            <td>${{esc(pos)}}</td>
            <td><code>${{esc(fn)}}</code> <span class="ok">${{esc(L.ok)}}</span></td>
            <td>
              <div class="row">
                <form method="post" action="/playlist/move">
                  <input type="hidden" name="playlist_id" value="${{esc(playlistId)}}" />
                  <input type="hidden" name="filename" value="${{esc(fn)}}" />
                  <input type="hidden" name="dir" value="up" />
                  <input type="hidden" name="q" value="${{esc(qVal)}}" />
                  <input type="hidden" name="lang" value="${{esc(langVal)}}" />
                  <button class="icon" data-move="up" type="submit"${{upDisabled}} title="${{esc(L.up)}}">&uarr;</button>
                </form>
                <form method="post" action="/playlist/move">
                  <input type="hidden" name="playlist_id" value="${{esc(playlistId)}}" />
                  <input type="hidden" name="filename" value="${{esc(fn)}}" />
                  <input type="hidden" name="dir" value="down" />
                  <input type="hidden" name="q" value="${{esc(qVal)}}" />
                  <input type="hidden" name="lang" value="${{esc(langVal)}}" />
                  <button class="icon" data-move="down" type="submit" disabled title="${{esc(L.down)}}">&darr;</button>
                </form>
              </div>
            </td>
            <td>
              <form method="post" action="/playlist/remove" class="row">
                <input type="hidden" name="playlist_id" value="${{esc(playlistId)}}" />
                <input type="hidden" name="filename" value="${{esc(fn)}}" />
                <input type="hidden" name="q" value="${{esc(qVal)}}" />
                <input type="hidden" name="lang" value="${{esc(langVal)}}" />
                <button type="submit">${{esc(L.exclude)}}</button>
              </form>
            </td>
            <td>
              <div class="row">
                <form method="post" action="/playlist/play" class="row">
                  <input type="hidden" name="playlist_id" value="${{esc(playlistId)}}" />
                  <input type="hidden" name="filename" value="${{esc(fn)}}" />
                  <input type="hidden" name="q" value="${{esc(qVal)}}" />
                  <input type="hidden" name="lang" value="${{esc(langVal)}}" />
                  <button type="submit">${{esc(L.play)}}</button>
                </form>
                <audio controls preload="none" src="/mp3/${{fnUrl}}"></audio>
              </div>
            </td>
          </tr>`;

        const tmp = document.createElement('tbody');
        tmp.innerHTML = rowHtml.trim();
        const newRow = tmp.firstElementChild;
        if (newRow) {{
          // If the included table was "empty" placeholder row, replace it.
          const placeholder = includedBody.querySelector('tr td.muted');
          if (placeholder && placeholder.closest('tr') && includedBody.children.length === 1) {{
            includedBody.innerHTML = '';
          }}
          includedBody.appendChild(newRow);
        }}
      }}

      showMsg(L.added, false);
    }} catch (e) {{
      showMsg(e && e.message ? e.message : L.generic_error, true);
    }} finally {{
      if (btn) btn.disabled = false;
    }}
  }});
}})();
</script>
"""

    back_href = redirect_location("/", {"lang": safe_lang})
    lang_switcher = html_lang_switcher(safe_lang, "/playlist", {"id": str(pid), "q": search})
    draft_hint_html = t(lang, "draft_hint", play_btn=html.escape(t(lang, "btn_play")))

    body = f"""
<header>
  <div>
    <h1>{html.escape(t(lang, "playlist_title", name=name))}</h1>
    <div class="muted"><a href="{html.escape(back_href, quote=True)}">{html.escape(t(lang, "nav_back"))}</a> | {html.escape(t(lang, "nav_stream"))}: <a href="/stream.mp3?playlist_id={pid}&loop=1">/stream.mp3</a> | {html.escape(t(lang, "nav_export"))}: <a href="/export.m3u?playlist_id={pid}">.m3u</a> | {html.escape(t(lang, "nav_id"))}: <code>{pid}</code></div>
    <div class="row" style="margin-top: 10px;">
      <audio controls preload="none" src="/stream.mp3?playlist_id={pid}&loop=1"></audio>
      <a class="pill" href="/stream.mp3?playlist_id={pid}&loop=1">loop=1</a>
      <a class="pill" href="/stream.mp3?playlist_id={pid}&loop=0">loop=0</a>
      {draft_label}
    </div>
  </div>
  <div class="row">
    {lang_switcher}
    <form method="get" action="/playlist" class="row">
      <input type="hidden" name="id" value="{pid}" />
      <input type="hidden" name="lang" value="{html.escape(safe_lang)}" />
      <input type="search" name="q" value="{html.escape(search)}" placeholder="{html.escape(t(lang, "playlist_search_placeholder"))}" />
      <button type="submit">{html.escape(t(lang, "btn_search"))}</button>
    </form>
  </div>
</header>
{msg_html}

<h2>{html.escape(t(lang, "section_draft"))}</h2>
<form method="post" action="/playlist/draft" class="row">
  <input type="hidden" name="playlist_id" value="{pid}" />
  <input type="hidden" name="enabled" value="{draft_next}" />
  <input type="hidden" name="q" value="{html.escape(search)}" />
  <input type="hidden" name="lang" value="{html.escape(safe_lang)}" />
  <button type="submit">{html.escape(draft_btn_label)}</button>
  <span class="muted">{draft_hint_html}</span>
</form>
<div class="muted">{draft_current_line}</div>

<h2>{html.escape(t(lang, "section_rename"))}</h2>
<form method="post" action="/playlist/rename" class="row">
  <input type="hidden" name="id" value="{pid}" />
  <input type="hidden" name="lang" value="{html.escape(safe_lang)}" />
  <input type="text" name="name" value="{html.escape(name)}" required />
  <button type="submit">{html.escape(t(lang, "btn_rename"))}</button>
</form>

<h2>{html.escape(t(lang, "section_included"))}</h2>
{included_table}

<h2>{html.escape(t(lang, "section_add_quick"))}</h2>
<div class="row">
  <input id="quickFilter" type="search" placeholder="{html.escape(t(lang, "placeholder_quick_filter"))}" />
  <span id="quickCount" class="muted"></span>
</div>
<div id="jsMsg" class="msg" style="display:none;"></div>
{excluded_table}
{js}
"""
    return html_page(lang, t(lang, "playlist_title", name=name), body)


def parse_post_form(handler: BaseHTTPRequestHandler) -> dict[str, list[str]]:
    ctype = handler.headers.get("Content-Type", "")
    if "application/x-www-form-urlencoded" not in ctype:
        return {}
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except Exception:
        length = 0
    raw = handler.rfile.read(length) if length > 0 else b""
    text = raw.decode("utf-8", errors="replace")
    return parse_qs(text, keep_blank_values=True)


RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "PlaylistWeb/1.0"
    protocol_version = "HTTP/1.1"

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError):
            # Streaming players / browsers may drop connections abruptly.
            return

    def _cookie_get(self, key: str) -> str | None:
        raw = self.headers.get("Cookie", "")
        if not raw:
            return None
        for part in raw.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            if k.strip() == key:
                return v.strip()
        return None

    def _resolve_lang(self, requested: str) -> str:
        req = (requested or "").strip().lower()
        if req in SUPPORTED_LANGS:
            self._set_lang_cookie = req
            return req
        ck = self._cookie_get("lang")
        if ck:
            ck2 = ck.strip().lower()
            if ck2 in SUPPORTED_LANGS:
                return ck2
        return DEFAULT_LANG

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        lang = self._resolve_lang(qp_first(qs, "lang", ""))

        if path == "/":
            q = qp_first(qs, "q", "")
            msg = qp_first(qs, "msg", "")
            err = qp_first(qs, "err", "")
            data = html_home(lang, q, msg, err)
            self._send_html(data)
            return

        if path == "/playlist":
            pid = as_int(qp_first(qs, "id", "0"), 0)
            q = qp_first(qs, "q", "")
            msg = qp_first(qs, "msg", "")
            err = qp_first(qs, "err", "")
            data = html_playlist(lang, pid, q, msg, err)
            self._send_html(data)
            return

        if path == "/export.m3u":
            pid = as_int(qp_first(qs, "playlist_id", "0"), 0)
            pl = db_get_playlist(pid)
            if not pl:
                self._send_text(t(lang, "err_playlist_not_found"), status=HTTPStatus.NOT_FOUND)
                return
            _pid, name = pl
            tracks = db_list_playlist_tracks(pid, "")
            # Standard M3U format: one filename per line (relative paths).
            lines = ["#EXTM3U"]
            for (fn, _pos) in tracks:
                lines.append(fn)
            payload = ("\n".join(lines) + "\n").encode("utf-8", errors="strict")
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or f"playlist_{pid}"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/x-mpegurl; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{safe}.m3u"')
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self._maybe_set_lang_cookie()
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/stream.mp3":
            pid = as_int(qp_first(qs, "playlist_id", "0"), 0)
            loop = as_int(qp_first(qs, "loop", "1"), 1)
            self._stream_playlist_mp3(pid, loop=bool(loop), lang=lang)
            return

        if path.startswith("/mp3/"):
            # Serve MP3 files from BASE_DIR only.
            name = unquote(path[len("/mp3/") :])
            ok, err = validate_mp3_name(name, lang)
            if not ok:
                self._send_text(err, status=HTTPStatus.BAD_REQUEST)
                return
            file_path = BASE_DIR / name
            self._send_file_range(file_path, content_type="audio/mpeg")
            return

        self._send_text(t(lang, "http_not_found"), status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        form = parse_post_form(self)

        def f(key: str) -> str:
            return (form.get(key) or [""])[0]

        lang = self._resolve_lang(f("lang"))
        safe_lang = normalize_lang(lang)

        def redirect_playlist(pid: int, msg: str = "", err: str = "") -> None:
            q = f("q").strip()
            params: dict[str, str] = {"id": str(pid), "lang": safe_lang}
            if q:
                params["q"] = q
            if msg:
                params["msg"] = msg
            if err:
                params["err"] = err
            self._redirect("/playlist", params)

        if path == "/api/playlist/add":
            pid = as_int(f("playlist_id"), 0)
            filename = f("filename")
            ok, err = db_add_track(pid, filename, lang)
            if not ok:
                self._send_json({"ok": False, "error": err}, status=HTTPStatus.BAD_REQUEST)
                return
            with db_connect() as conn:
                row = conn.execute(
                    "SELECT position FROM playlist_items WHERE playlist_id = ? AND filename = ?;",
                    (pid, filename),
                ).fetchone()
            pos = int(row["position"]) if row else 0
            self._send_json({"ok": True, "filename": filename, "position": pos})
            return

        if path == "/playlist/create":
            ok, err = db_create_playlist(f("name"), lang)
            if ok:
                self._redirect("/", {"msg": t(lang, "msg_playlist_created"), "lang": safe_lang})
            else:
                self._redirect("/", {"err": err, "lang": safe_lang})
            return

        if path == "/playlist/delete":
            pid = as_int(f("id"), 0)
            if pid > 0:
                db_delete_playlist(pid)
            self._redirect("/", {"msg": t(lang, "msg_playlist_deleted"), "lang": safe_lang})
            return

        if path == "/playlist/rename":
            pid = as_int(f("id"), 0)
            ok, err = db_rename_playlist(pid, f("name"), lang)
            if ok:
                redirect_playlist(pid, msg=t(lang, "msg_playlist_renamed"))
            else:
                redirect_playlist(pid, err=err)
            return

        if path == "/playlist/add":
            pid = as_int(f("playlist_id"), 0)
            filename = f("filename")
            ok, err = db_add_track(pid, filename, lang)
            if ok:
                redirect_playlist(pid, msg=t(lang, "msg_track_added"))
            else:
                redirect_playlist(pid, err=err)
            return

        if path == "/playlist/remove":
            pid = as_int(f("playlist_id"), 0)
            filename = f("filename")
            db_remove_track(pid, filename)
            redirect_playlist(pid, msg=t(lang, "msg_playlist_changed"))
            return

        if path == "/playlist/move":
            pid = as_int(f("playlist_id"), 0)
            filename = f("filename")
            dir_s = f("dir").strip().lower()
            direction = -1 if dir_s == "up" else (1 if dir_s == "down" else 0)
            ok, err = db_move_track(pid, filename, direction, lang)
            if ok:
                redirect_playlist(pid, msg=t(lang, "msg_order_changed"))
            else:
                redirect_playlist(pid, err=err)
            return

        if path == "/playlist/draft":
            pid = as_int(f("playlist_id"), 0)
            enabled = bool(as_int(f("enabled"), 0))
            ok, err = db_set_draft_mode(pid, enabled, lang)
            if ok:
                if enabled:
                    redirect_playlist(pid, msg=t(lang, "msg_draft_enabled", play_btn=t(lang, "btn_play")))
                else:
                    redirect_playlist(pid, msg=t(lang, "msg_draft_disabled"))
            else:
                redirect_playlist(pid, err=err)
            return

        if path == "/playlist/play":
            pid = as_int(f("playlist_id"), 0)
            filename = f("filename")
            # Pragmatic: clicking "Play" means "send this to the stream", so also enable draft mode.
            ok, err = db_set_stream_track(pid, filename, enable_draft=True, lang=lang)
            if ok:
                redirect_playlist(pid, msg=t(lang, "msg_stream_set", filename=filename))
            else:
                redirect_playlist(pid, err=err)
            return

        self._send_text(t(lang, "http_not_found"), status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs short.
        msg = fmt % args
        print(f"[{utc_now_iso()}] {self.address_string()} {self.command} {self.path} - {msg}")

    def _maybe_set_lang_cookie(self) -> None:
        lang = getattr(self, "_set_lang_cookie", "")
        if not lang:
            return
        # Persist language selection for 1 year.
        self.send_header("Set-Cookie", f"lang={lang}; Path=/; Max-Age=31536000; SameSite=Lax")

    def _send_html(self, data: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = text.encode("utf-8", errors="strict")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8", errors="strict")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, path: str, qs: dict[str, str] | None = None) -> None:
        loc = redirect_location(path, qs)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()

    def _send_file_range(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = size - 1
        status = HTTPStatus.OK

        if range_header:
            m = RANGE_RE.match(range_header.strip())
            if m:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start <= end:
                    status = HTTPStatus.PARTIAL_CONTENT

        length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()

        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(FILE_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _stream_looping_mp3(self, pid: int) -> None:
        # Single long-lived MP3 stream that can switch between:
        # - normal mode: plays the playlist in order (looping)
        # - draft mode: plays ONLY the currently selected track (looping that track)
        #
        # This allows a connected streaming player to react without reconnecting.
        conn = db_connect()
        try:
            conn.execute("INSERT OR IGNORE INTO playlist_stream_state(playlist_id) VALUES (?);", (pid,))
            conn.commit()

            def get_state() -> tuple[bool, str | None, int] | None:
                row = conn.execute(
                    "SELECT draft_mode, current_filename, version FROM playlist_stream_state WHERE playlist_id = ?;",
                    (pid,),
                ).fetchone()
                if not row:
                    return None
                draft_on = bool(int(row["draft_mode"]))
                cur = str(row["current_filename"]) if row["current_filename"] is not None else None
                ver = int(row["version"])
                return draft_on, cur, ver

            def get_playlist_filenames() -> list[str]:
                rows = conn.execute(
                    "SELECT filename FROM playlist_items WHERE playlist_id = ? ORDER BY position ASC;",
                    (pid,),
                ).fetchall()
                out: list[str] = []
                for r in rows:
                    fn = str(r["filename"])
                    ok, _err = validate_mp3_name(fn)
                    if ok:
                        out.append(fn)
                return out

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "close",
            }

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self._maybe_set_lang_cookie()
            self.close_connection = True
            self.end_headers()

            def send_chunk(data: bytes) -> None:
                if not data:
                    return
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            bitrate_cache: dict[str, int] = {}

            def bytes_per_second_for(fn: str) -> float:
                kbps = bitrate_cache.get(fn)
                if kbps is None:
                    kbps = mp3_bitrate_kbps(BASE_DIR / fn) or 160
                    bitrate_cache[fn] = kbps
                return (kbps * 1000) / 8.0

            mode: str | None = None  # "draft" or "normal"
            draft_fn: str | None = None
            draft_ver: int | None = None
            playlist: list[str] = []
            playlist_idx = 0

            f = None
            limiter: RateLimiter | None = None
            current_fn: str | None = None

            while True:
                st = get_state()
                if not st:
                    return
                draft_on, sel_fn, ver = st

                if draft_on:
                    # Draft mode: require a selected track. If none yet, keep the connection open and wait.
                    if not sel_fn or not validate_mp3_name(sel_fn)[0]:
                        time.sleep(0.2)
                        continue

                    if mode != "draft" or draft_fn != sel_fn or draft_ver != ver:
                        mode = "draft"
                        draft_fn = sel_fn
                        draft_ver = ver
                        playlist = []
                        playlist_idx = 0
                        current_fn = sel_fn
                        if f is not None:
                            try:
                                f.close()
                            except Exception:
                                pass
                            f = None
                        try:
                            f = (BASE_DIR / sel_fn).open("rb")
                        except FileNotFoundError:
                            time.sleep(0.2)
                            continue
                        limiter = RateLimiter(bytes_per_second_for(sel_fn), burst_seconds=0.1)

                    assert f is not None
                    assert current_fn is not None
                    buf = f.read(DRAFT_STREAM_CHUNK_SIZE)
                    if not buf:
                        f.seek(0)
                        continue
                    send_chunk(buf)
                    if limiter:
                        limiter.on_send(len(buf))
                    continue

                # Normal mode: play playlist in order, looping.
                if mode != "normal":
                    mode = "normal"
                    draft_fn = None
                    draft_ver = None
                    current_fn = None
                    if f is not None:
                        try:
                            f.close()
                        except Exception:
                            pass
                        f = None
                    limiter = None
                    playlist = []
                    playlist_idx = 0

                if not playlist or playlist_idx >= len(playlist):
                    playlist = get_playlist_filenames()
                    playlist_idx = 0
                    if not playlist:
                        time.sleep(0.5)
                        continue

                if f is None:
                    current_fn = playlist[playlist_idx]
                    try:
                        f = (BASE_DIR / current_fn).open("rb")
                    except FileNotFoundError:
                        playlist_idx += 1
                        current_fn = None
                        continue
                    limiter = RateLimiter(bytes_per_second_for(current_fn), burst_seconds=0.25)

                buf = f.read(STREAM_CHUNK_SIZE)
                if not buf:
                    try:
                        f.close()
                    except Exception:
                        pass
                    f = None
                    limiter = None
                    current_fn = None
                    playlist_idx += 1
                    continue

                send_chunk(buf)
                if limiter:
                    limiter.on_send(len(buf))

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

    def _stream_draft_mp3(self, pid: int, loop: bool, lang: str = DEFAULT_LANG) -> None:
        # In draft mode the stream plays ONLY the last clicked track.
        conn = db_connect()
        try:
            def get_state() -> tuple[bool, str | None, int] | None:
                row = conn.execute(
                    "SELECT draft_mode, current_filename, version FROM playlist_stream_state WHERE playlist_id = ?;",
                    (pid,),
                ).fetchone()
                if not row:
                    return None
                return bool(int(row["draft_mode"])), (str(row["current_filename"]) if row["current_filename"] is not None else None), int(
                    row["version"]
                )

            st = get_state()
            if not st:
                self._send_text(t(lang, "err_playlist_not_found"), status=HTTPStatus.NOT_FOUND)
                return
            draft_on, cur_fn, cur_ver = st
            if not draft_on:
                self._send_text(t(lang, "err_draft_disabled"), status=HTTPStatus.NOT_FOUND)
                return
            if not cur_fn:
                self._send_text(t(lang, "err_draft_no_selection", play_btn=t(lang, "btn_play")), status=HTTPStatus.NOT_FOUND)
                return
            ok, err = validate_mp3_name(cur_fn, lang)
            if not ok:
                self._send_text(t(lang, "err_draft_prefix", err=err), status=HTTPStatus.NOT_FOUND)
                return

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "close",
            }

            if not loop:
                p = BASE_DIR / cur_fn
                size = p.stat().st_size
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/mpeg")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(size))
                self._maybe_set_lang_cookie()
                self.close_connection = True
                self.end_headers()

                try:
                    with p.open("rb") as f:
                        while True:
                            chunk = f.read(FILE_CHUNK_SIZE)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, FileNotFoundError):
                    return
                return

            # loop=1: chunked + switch on version changes
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self._maybe_set_lang_cookie()
            self.close_connection = True
            self.end_headers()

            def send_chunk(data: bytes) -> None:
                if not data:
                    return
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")

            active_fn = cur_fn
            active_ver = cur_ver

            while True:
                p = BASE_DIR / active_fn
                bitrate_kbps = mp3_bitrate_kbps(p) or 160
                # Pace near real-time to avoid huge client buffers; this is key for instant switching.
                limiter = RateLimiter((bitrate_kbps * 1000) / 8.0, burst_seconds=0.1)
                try:
                    with p.open("rb") as f:
                        while True:
                            # Check for track switch BEFORE sending more bytes, so changes apply immediately.
                            st2 = get_state()
                            if not st2:
                                return
                            draft_on2, fn2, ver2 = st2
                            if not draft_on2:
                                return
                            if ver2 != active_ver:
                                if fn2 and validate_mp3_name(fn2)[0]:
                                    active_fn = fn2
                                    active_ver = ver2
                                    break
                                active_ver = ver2

                            buf = f.read(DRAFT_STREAM_CHUNK_SIZE)
                            if not buf:
                                break
                            send_chunk(buf)
                            limiter.on_send(len(buf))
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                except FileNotFoundError:
                    # If the file disappeared, wait for a new selection (client will likely reconnect anyway).
                    return

                # If we broke out due to version change, start the new file immediately.
                st3 = get_state()
                if not st3:
                    return
                draft_on3, fn3, ver3 = st3
                if not draft_on3:
                    return
                if ver3 != active_ver and fn3 and validate_mp3_name(fn3)[0]:
                    active_fn = fn3
                    active_ver = ver3
                    continue
                # Otherwise repeat current file forever.

        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _stream_playlist_mp3(self, pid: int, loop: bool, lang: str = DEFAULT_LANG) -> None:
        pl = db_get_playlist(pid)
        if not pl:
            self._send_text(t(lang, "err_playlist_not_found"), status=HTTPStatus.NOT_FOUND)
            return

        if loop:
            self._stream_looping_mp3(pid)
            return

        stream_state = db_get_stream_state(pid)
        if stream_state and stream_state.draft_mode:
            self._stream_draft_mp3(pid, loop=False, lang=lang)
            return

        tracks = db_list_playlist_tracks(pid, "")
        files: list[Path] = []
        for (fn, _pos) in tracks:
            ok, _err = validate_mp3_name(fn)
            if ok:
                files.append(BASE_DIR / fn)

        if not files:
            self._send_text(t(lang, "err_playlist_no_valid_mp3"), status=HTTPStatus.NOT_FOUND)
            return

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "close",
        }

        total = 0
        for p in files:
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/mpeg")
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(total))
        self._maybe_set_lang_cookie()
        self.close_connection = True
        self.end_headers()

        try:
            for p in files:
                try:
                    with p.open("rb") as f:
                        while True:
                            chunk = f.read(FILE_CHUNK_SIZE)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except FileNotFoundError:
                    continue
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple MP3 playlist manager (HTML + SQLite, no deps) / Prosty manager playlist MP3 (HTML + SQLite, bez zaleznosci)."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1) / Adres do bindowania (domyslnie: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001) / Port (domyslnie: 8001).")
    args = parser.parse_args()

    db_init()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[{utc_now_iso()}] Serving on http://{args.host}:{args.port}/  (folder: {BASE_DIR})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{utc_now_iso()}] Shutting down...")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
