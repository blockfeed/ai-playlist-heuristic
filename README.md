# ai-playlist.py (totally-not-AI™)

*A tongue-in-cheek “AI” playlist generator.*  
In reality it’s **TF-IDF text matching + Gaussian BPM scoring + hand-rolled heuristics**.  
No model training, no cloud, no adaptation. Just reproducible math that often makes **good starting playlists**.

> ⚠️ **Status**: experimental, imperfect, and intentionally modest.  
> Think of this as a nerdy shuffler with opinions — not Spotify’s brain.

---

## What it does

- Scans your music library (FLAC/MP3/OPUS/etc.).
- Scores tracks against your **prompt** + a **preset**:
  - **study** (~85 BPM, vocals discouraged)
  - **running** (~168 BPM, tempo-driven)
  - **sleep** (~60 BPM, super low energy)
  - **romantic** (~88 BPM, warm/jazz/soul)
  - **dance_party** (~124 BPM, club tempo)
- Writes **M3U** or **XSPF** playlists with relative paths (Rockbox-friendly).

> Optional: estimates BPM with `librosa` if your tags don’t have it.

---

## Install (Arch Linux)

```bash
sudo pacman -S python-mutagen python-scikit-learn python-numpy                python-librosa python-soundfile
```

`librosa` is optional — enable analysis only if you pass `--analyze-tempo`.

---

## Quick start

```bash
# Study (M3U)
python ./ai_playlist.py --music-dir "$HOME/Music"   --prompt "study: calm, instrumental, ambient"   --preset study --format m3u   --out "$HOME/Playlists/Study.m3u"

# Running (M3U)
python ./ai_playlist.py --music-dir "$HOME/Music"   --prompt "tempo for 8k run"   --preset running --format m3u   --out "$HOME/Playlists/Run.m3u"

# Sleep (XSPF)
python ./ai_playlist.py --music-dir "$HOME/Music"   --prompt "sleep, drones, soft piano"   --preset sleep --format xspf   --out "$HOME/Playlists/Sleep.xspf"

# Dance party with pairings bias (see below)
python ./ai_playlist.py --music-dir "$HOME/Music"   --prompt "party, upbeat"   --preset dance_party --format m3u   --pairings "$HOME/pairings.json"   --out "$HOME/Playlists/Party.m3u"
```

---

## How it Works

1. **Library Scan**  
   - Recursively walks `--music-dir`, collecting supported audio files.  
   - Reads tags via `mutagen`: `title, artist, album, albumartist, genre, comment, lyrics, bpm, duration`.  
   - If `--analyze-tempo` is enabled, estimates BPM from audio with `librosa`.

2. **Text Modeling (TF-IDF)**  
   - Each track’s tags are concatenated into a “document.”  
   - `scikit-learn` `TfidfVectorizer` scores relevance against the `--prompt` plus preset “prompt_enrich” hints.  
   - Falls back to keyword overlap if scikit-learn isn’t installed.  
   - Scores normalized to [0,1].

3. **Tempo Scoring**  
   - Compares BPM (tag or estimated) to a preset Gaussian target (e.g., **running** 168 ± 18 BPM).  
   - Produces a [0,1] “tempo fit.”

4. **Penalty/Bonus Heuristics**  
   - Penalizes: *live*, *remix/remaster/radio edit*, *feat.*, *lyrics present*, *vocal* genres (weights vary by preset).  
   - Keyword bonuses for preset “positives” (e.g., *ambient* for sleep) and negatives for preset “anti-genres”.

5. **Final Score**  
   ```
   final = (w_text * text_score)
         + (w_tempo * tempo_score)
         - (w_pen   * penalty)
         + bonus
   ```
   - Weights tuned per preset (tempo matters more for **running**/**dance_party**).

6. **Selection**  
   - Sorted by score, deterministic tie-break with BLAKE2b (`--seed`).  
   - Per-artist cap enforced (`--artist-max`).  
   - Optional `--pairings` JSON boosts related tracks you like.

7. **Output**  
   - Writes `.m3u` (with `#EXTINF`) or `.xspf` (XML).  
   - Paths are relative to the playlist directory unless `--relative-to` is given.  
   - `--dry-run` prints top candidates with score/BPM.

---

## Pairings / Affinities (optional)

You can nudge selections toward **your** known pairings (no secret training data).  
Create a file like `pairings.json`:

```json
{
  "rules": [
    {
      "if":    { "artist": "Daft Punk" },
      "boost": [ { "artist": "Justice" }, { "title_contains": "Remix" } ]
    },
    {
      "if":    { "artist": "Nina Simone" },
      "boost": [ { "artist": "Billie Holiday" }, { "artist": "Etta James" } ]
    }
  ]
}
```

Then run with `--pairings /path/to/pairings.json`.

---

## Limitations (a.k.a. honesty corner)

- **Not actual AI**: no model training or adaptation.  
- **Garbage in → garbage out**: messy tags = messy results.  
- **Tempo analysis** is slow; prefer BPM tags when possible.  
- **Heuristics are opinionated**; you may need to hand-tune results.

---

## License

This project is licensed under the **GNU GPLv3**. See [LICENSE](LICENSE) for details.
