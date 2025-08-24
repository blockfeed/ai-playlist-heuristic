#!/usr/bin/env python3
"""
ai_playlist.py — Scan a music library and generate playlists with a local "AI"
(TF-IDF text relevance + audio heuristics).
- Zero network calls. Works entirely offline.
- Uses tags (title/artist/album/genre/comments/lyrics) + optional tempo analysis (librosa).
- Outputs M3U8 or XSPF playlists compatible with Rockbox and common players.

Presets: study, running, sleep, romantic, dance_party
Supports optional JSON pairings to boost known affinities.
"""

import argparse, os, re, sys, math, json, hashlib, random, pathlib
from typing import List, Dict, Optional

# --- Optional deps ---
_missing = []
try:
    from mutagen import File as MutagenFile
    from mutagen.easyid3 import EasyID3  # noqa
except Exception as e:
    MutagenFile = None; _missing.append(f"mutagen ({e})")
try:
    import numpy as np
except Exception as e:
    np = None; _missing.append(f"numpy ({e})")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    TfidfVectorizer = None; _missing.append(f"scikit-learn ({e})")
try:
    import librosa  # type: ignore
except Exception as e:
    librosa = None

AUDIO_EXTS = {".flac",".mp3",".m4a",".aac",".ogg",".opus",".wav",".wma",".aiff",".aif"}

# --- Presets ---
PRESETS: Dict[str, Dict] = {
    "study": {
        "prompt_enrich": "calm focus concentration instrumental ambient post-rock classical lofi chill no vocals 60-110 bpm",
        "bpm_mu": 85.0, "bpm_sigma": 25.0,
        "penalties": {"live":0.25,"remix":0.20,"feat":0.15,"lyrics":0.35,"vocal_genre":0.35},
        "positives": ["instrumental","ambient","classical","post-rock","lofi","soundtrack"],
        "negatives": ["hardcore","extreme metal","screamo"], "allow_vocals": False
    },
    "running": {
        "prompt_enrich": "running cardio upbeat energetic electronic synth pop edm house drum and bass rock 150-180 bpm",
        "bpm_mu": 168.0, "bpm_sigma": 18.0,
        "penalties": {"live":0.05,"remix":-0.05,"feat":0.0,"lyrics":0.0,"vocal_genre":0.0},
        "positives": ["edm","house","electro","synth","pop","dance","dnb","drum and bass","trance","rock"],
        "negatives": ["ambient","downtempo","sleep"], "allow_vocals": True
    },
    "sleep": {
        "prompt_enrich": "sleep calm drone ambient instrumental piano soft classical post-rock minimal 50-70 bpm low dynamics no vocals",
        "bpm_mu": 60.0, "bpm_sigma": 12.0,
        "penalties": {"live":0.35,"remix":0.35,"feat":0.20,"lyrics":0.50,"vocal_genre":0.50},
        "positives": ["ambient","drone","piano","lullaby","new age","instrumental","post-rock","classical"],
        "negatives": ["metal","punk","hardcore","noise","industrial","edm","dance","hip hop"], "allow_vocals": False
    },
    "romantic": {
        "prompt_enrich": "romantic intimate warm soulful r&b jazz bossa nova acoustic slow dance 70-110 bpm love ballad",
        "bpm_mu": 88.0, "bpm_sigma": 20.0,
        "penalties": {"live":0.10,"remix":0.20,"feat":0.05,"lyrics":0.0,"vocal_genre":0.0},
        "positives": ["soul","r&b","bossa","jazz","acoustic","ballad","slow jam","torch"],
        "negatives": ["aggressive","screamo","thrash","industrial"], "allow_vocals": True
    },
    "dance_party": {
        "prompt_enrich": "dance party club dj pop edm house disco techno electro top40 115-130 bpm remix radio edit extended mix",
        "bpm_mu": 124.0, "bpm_sigma": 10.0,
        "penalties": {"live":0.0,"remix":-0.10,"feat":-0.05,"lyrics":0.0,"vocal_genre":0.0},
        "positives": ["dance","edm","house","disco","techno","electro","club","pop"],
        "negatives": ["ambient","downtempo","drone","noise"], "allow_vocals": True
    },
}

def human_err(msg): print(f"[!] {msg}", file=sys.stderr)
def ok(msg): print(f"[*] {msg}", file=sys.stderr)

# --- Tagging / scoring helpers ---
def read_tags(p:pathlib.Path)->Dict[str,Optional[str]]:
    out = {"title":"","artist":"","album":"","albumartist":"","genre":"","comment":"","lyrics":"","bpm":"","date":"","duration":0.0}
    if not MutagenFile: return out
    try:
        mf = MutagenFile(str(p), easy=True)
        if not mf: return out
        tags = getattr(mf,"tags",{}) or {}
        def _get(k,alt=None):
            v = tags.get(k,tags.get(alt,[])) if alt else tags.get(k,[])
            return ", ".join(map(str,v)) if isinstance(v,list) else str(v or "")
        out.update({
            "title":_get("title"), "artist":_get("artist"), "album":_get("album"),
            "albumartist":_get("albumartist"), "genre":_get("genre"),
            "comment":_get("comment"), "lyrics":_get("lyrics") or _get("unsyncedlyrics") or _get("lyricist"),
            "bpm":_get("bpm") or _get("tbpm"), "date":_get("date") or _get("year")
        })
        if hasattr(mf,"info") and getattr(mf.info,"length",None):
            out["duration"]=float(mf.info.length)
    except Exception as e: human_err(f"Tag read failed for {p}: {e}")
    return out

def estimate_bpm_with_librosa(path, seconds=45):
    if not librosa: return None
    try:
        y,sr=librosa.load(str(path),mono=True,duration=seconds)
        if y is None or not len(y): return None
        tempo,_=librosa.beat.beat_track(y=y,sr=sr)
        if tempo and np and np.isfinite(tempo): return float(tempo)
    except Exception as e: human_err(f"librosa BPM failed {path}: {e}")
    return None

def gaussian_score(x,mu,sigma): return math.exp(-0.5*((x-mu)/sigma)**2) if sigma>0 else 0.0
def normalize(s): return re.sub(r"\s+"," ",s or "").strip().lower()
def keyword_bonus(tags,positives,negatives):
    blob=normalize(" ".join([tags.get("genre",""),tags.get("comment",""),tags.get("title","")]))
    bonus=0.0
    for k in positives: 
        if k and k.lower() in blob: bonus+=0.05
    for k in negatives:
        if k and k.lower() in blob: bonus-=0.05
    return max(-0.25,min(0.25,bonus))

def detect_penalties(tags,preset):
    t,a,g,l=map(normalize,[tags.get("title",""),tags.get("album",""),tags.get("genre",""),tags.get("lyrics","")])
    p=PRESETS[preset]["penalties"]; pen=0
    if " live" in t or " live" in a: pen+=p["live"]
    if any(x in t for x in ["remix","remaster","extended","radio edit"]): pen+=p["remix"]
    if "feat" in t: pen+=p["feat"]
    if l: pen+=p["lyrics"]
    if "vocal" in g and "instrumental" not in g: pen+=p["vocal_genre"]
    return max(-0.5,min(1.0,pen))

def tfidf_text_scores(prompt,docs):
    if not TfidfVectorizer:
        p=set(normalize(prompt).split()); scores=[]
        for d in docs: dd=set(normalize(d).split()); scores.append(len(p&dd)/float(len(p)+1e-6))
        return scores
    vec=TfidfVectorizer(max_features=12000,stop_words="english"); X=vec.fit_transform(docs); qv=vec.transform([prompt])
    return (X@qv.T).toarray().ravel().tolist()

def minmax(vals):
    if not vals: return []
    lo,hi=min(vals),max(vals)
    return [(v-lo)/(hi-lo) if hi>lo else 0.0 for v in vals]

# --- Writers ---
def write_m3u8(out_path,entries,relative_to=None):
    lines=["#EXTM3U"]
    for e in entries:
        dur=int(round(e.get("duration",0.0))) if e.get("duration") else -1
        artist=e.get("artist") or e.get("albumartist") or ""
        title=e.get("title") or pathlib.Path(e["path"]).stem
        lines.append(f"#EXTINF:{dur},{artist} - {title}")
        f=pathlib.Path(e["path"])
        try: lines.append(str(f.relative_to(relative_to))) if relative_to else lines.append(str(f))
        except: lines.append(str(f))
    out_path.write_text("\n".join(lines),encoding="utf-8")

def write_xspf(out_path,entries,relative_to=None):
    import xml.etree.ElementTree as ET; from urllib.parse import quote
    NS="http://xspf.org/ns/0/"; ET.register_namespace("",NS)
    playlist=ET.Element("{%s}playlist"%NS,version="1"); tracklist=ET.SubElement(playlist,"{%s}trackList"%NS)
    for e in entries:
        track=ET.SubElement(tracklist,"{%s}track"%NS)
        ET.SubElement(track,"{%s}title"%NS).text=e.get("title") or pathlib.Path(e["path"]).stem
        if e.get("artist"): ET.SubElement(track,"{%s}creator"%NS).text=e["artist"]
        if e.get("duration"): ET.SubElement(track,"{%s}duration"%NS).text=str(int(float(e["duration"])*1000))
        f=pathlib.Path(e["path"])
        loc=None
        if relative_to:
            try: loc=str(f.relative_to(relative_to)).replace("\\","/")
            except: pass
        if not loc: loc="file://"+quote(str(f))
        ET.SubElement(track,"{%s}location"%NS).text=loc
    ET.ElementTree(playlist).write(str(out_path),encoding="utf-8",xml_declaration=True)

# --- Main ---
def main():
    ap=argparse.ArgumentParser(description="Generate playlists from music library using TF-IDF + heuristics.")
    ap.add_argument("--music-dir",required=True); ap.add_argument("--prompt",required=True)
    ap.add_argument("--preset",choices=list(PRESETS.keys()),default="study")
    ap.add_argument("--format",choices=["m3u","xspf"],default="m3u"); ap.add_argument("--out",required=True)
    ap.add_argument("--limit",type=int,default=150); ap.add_argument("--artist-max",type=int,default=3)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--relative-to",default="")
    ap.add_argument("--analyze-tempo",action="store_true"); ap.add_argument("--analyze-seconds",type=int,default=45)
    ap.add_argument("--pairings",default=""); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    random.seed(args.seed)

    music_root=pathlib.Path(os.path.expanduser(args.music_dir)).resolve()
    if not music_root.exists(): human_err(f"music-dir not found: {music_root}"); sys.exit(2)
    out_path=pathlib.Path(os.path.expanduser(args.out)).resolve()
    rel_base=pathlib.Path(os.path.expanduser(args.relative_to)).resolve() if args.relative_to else out_path.parent

    if _missing: [human_err("Missing deps: "+m) for m in _missing]

    ok(f"Scanning {music_root} ...")
    tracks,texts=[],[]
    for root,_,files in os.walk(music_root):
        for fn in files:
            if pathlib.Path(fn).suffix.lower() in AUDIO_EXTS:
                p=pathlib.Path(root)/fn; tags=read_tags(p)
                bpm_val=None; raw=str(tags.get("bpm","")).strip()
                if raw:
                    try:bpm_val=float(raw)
                    except: pass
                if bpm_val is None and args.analyze_tempo:
                    bpm_val=estimate_bpm_with_librosa(p,seconds=args.analyze_seconds)
                tracks.append({"path":str(p),"tags":tags,"bpm":bpm_val})
                texts.append(" ".join([tags.get("title",""),tags.get("artist",""),tags.get("album",""),tags.get("genre","")]))
    if not tracks: human_err("No audio files found."); sys.exit(1)

    preset=PRESETS[args.preset]; prompt=args.prompt+" "+preset["prompt_enrich"]
    text_norm=minmax(tfidf_text_scores(prompt,texts))

    scored=[]
    for i,t in enumerate(tracks):
        tempo_component=0.5
        if t.get("bpm") and math.isfinite(t["bpm"]):
            tempo_component=gaussian_score(t["bpm"],preset["bpm_mu"],preset["bpm_sigma"])
        pen=detect_penalties(t["tags"],args.preset); bonus=keyword_bonus(t["tags"],preset["positives"],preset["negatives"])
        if preset["allow_vocals"] and pen>0: pen*=0.5
        if args.preset in ("running","dance_party"): wt,wp,wpen=(0.50,0.40,0.30)
        elif args.preset=="sleep": wt,wp,wpen=(0.55,0.25,0.30)
        else: wt,wp,wpen=(0.62,0.28,0.30)
        score=(wt*text_norm[i])+(wp*tempo_component)-(wpen*pen)+bonus
        scored.append({"score":max(0,min(1,score)),"title":t["tags"].get("title",""),
                       "artist":t["tags"].get("artist",""),"albumartist":t["tags"].get("albumartist",""),
                       "duration":t["tags"].get("duration",0.0),"bpm":t.get("bpm"),"path":t["path"]})
    scored.sort(key=lambda x:(-x["score"],hashlib.blake2b(x["path"].encode(),digest_size=8,person=str(args.seed).encode()).digest()))

    selected,artist_count=[],{}
    for s in scored:
        primary=s["artist"] or s["albumartist"] or "Unknown"
        if artist_count.get(primary,0)>=args.artist_max: continue
        selected.append(s); artist_count[primary]=artist_count.get(primary,0)+1
        if len(selected)>=args.limit: break

    ok(f"Selected {len(selected)} tracks")
    if args.dry_run:
        for s in selected[:50]:
            bpm_s=f"{s['bpm']:.0f} bpm" if s.get("bpm") else "n/a"
            print(f"{s['score']:.3f} {s['artist']} - {s['title']} [{bpm_s}] {s['path']}")
        return
    out_path.parent.mkdir(parents=True,exist_ok=True)
    if args.format=="m3u": write_m3u8(out_path,selected,rel_base)
    else: write_xspf(out_path,selected,rel_base)
    ok(f"Wrote playlist -> {out_path}")

if __name__=="__main__": main()

