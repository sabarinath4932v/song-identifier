import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import csv
import io
import os
from collections import defaultdict
from scipy.ndimage import maximum_filter

# ── Constants (UNCHANGED — core algorithm parameters) ──────────────────────────
DB_PATH = "fingerprint_database_light.csv"
SR      = 22050
N_FFT   = 2048
HOP     = 512
NEIGH   = 20
THRESH  = -25
FAN_OUT = 3
DT_MAX  = 60

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SOUNDPRINT",
    page_icon  = "◉",
    layout     = "wide"
)

# ── CSS — pure black + neon, Apple-keynote-style sleek sans ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:        #000000;
    --panel:     #0a0a0a;
    --panel-2:   #111111;
    --border:    #1f1f1f;
    --text:      #f2f2f7;
    --text-dim:  #6e6e76;
    --neon-cyan: #00f5ff;
    --neon-pink: #ff00e5;
    --neon-lime: #aef359;
    --neon-violet:#9d4dff;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
.block-container { padding-top: 1rem !important; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    padding: 4rem 0 3rem 0;
    text-align: center;
    position: relative;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.45em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    text-shadow: 0 0 12px rgba(0,245,255,0.55);
}
.hero-title {
    font-size: 5.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 0.95;
    margin: 0;
    background: linear-gradient(180deg, #ffffff 0%, #b9b9c4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title .accent {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet) 55%, var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--text-dim);
    margin-top: 1.4rem;
    max-width: 540px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* ── Tabs reimagined as a segmented pill nav ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border: 1px solid var(--border);
    border-radius: 999px;
    gap: 0.2rem;
    padding: 0.35rem;
    width: fit-content;
    margin: 0 auto 2.5rem auto;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    background: transparent !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.7rem 1.6rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: #000 !important;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet)) !important;
    box-shadow: 0 0 22px rgba(157,77,255,0.45);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { background: transparent !important; }

/* ── Section eyebrow / title block ── */
.sec-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    color: var(--neon-lime);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.sec-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0 0 1.6rem 0;
}

/* ── Glass / neon panel ── */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
}

/* ── Song cards (library grid) ── */
.song-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1rem 1.2rem;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.song-card:hover {
    border-color: var(--neon-cyan);
    box-shadow: 0 0 24px rgba(0,245,255,0.18);
    transform: translateY(-2px);
}
.card-name {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--text);
    margin: 0.6rem 0 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: var(--neon-lime);
}

/* ── Admin / empty state box ── */
.admin-box {
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: 18px;
    padding: 2.2rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 2rem;
    line-height: 1.9;
}

/* ── Match hero result ── */
.match-wrap {
    background: radial-gradient(circle at 15% 20%, rgba(0,245,255,0.10), transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(255,0,229,0.10), transparent 45%),
                var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 2.6rem 2.8rem;
    margin: 1.8rem 0;
    position: relative;
    overflow: hidden;
}
.match-wrap::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px; height: 3px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet), var(--neon-pink));
}
.match-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    color: var(--neon-pink);
    margin-bottom: 0.7rem;
    text-shadow: 0 0 10px rgba(255,0,229,0.5);
}
.match-song {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0;
    line-height: 1.1;
}
.match-stats {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 0.9rem;
}
.match-stats b { color: var(--neon-cyan); }

/* ── Candidate score bars ── */
.cand-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    color: var(--text-dim);
    margin: 1.8rem 0 0.9rem;
    text-transform: uppercase;
}
.score-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.score-name { color: var(--text); min-width: 230px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.score-track { flex: 1; background: var(--panel-2); border-radius: 3px; height: 6px; overflow: hidden; }
.score-fill-gold { background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet)); height: 6px; border-radius: 3px; box-shadow: 0 0 10px rgba(0,245,255,0.6); }
.score-fill-dim  { background: #2a2a30; height: 6px; border-radius: 3px; }
.score-num { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim); min-width: 50px; text-align: right; }
.score-num-top { color: var(--neon-cyan); }

/* ── Pipeline step blocks ── */
.step-wrap {
    border-left: 3px solid var(--neon-violet);
    padding-left: 1.4rem;
    margin: 3rem 0 1.2rem;
}
.step-eye {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    color: var(--neon-violet);
    margin-bottom: 0.35rem;
    text-transform: uppercase;
}
.step-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text); margin: 0; }
.step-desc { font-size: 0.88rem; color: var(--text-dim); margin-top: 0.5rem; line-height: 1.7; font-weight: 300; }
.step-desc b { color: var(--neon-cyan); font-weight: 600; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet));
    color: #000;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: none;
    border-radius: 999px;
    padding: 0.8rem 2.2rem;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 0 0 rgba(0,245,255,0);
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.015);
    box-shadow: 0 6px 28px rgba(157,77,255,0.45);
}
.stButton > button:active { transform: translateY(0) scale(0.99); }

.stDownloadButton > button {
    background: transparent;
    color: var(--neon-lime);
    border: 1px solid var(--neon-lime);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.7rem 1.8rem;
}
.stDownloadButton > button:hover {
    background: rgba(174,243,89,0.08);
    box-shadow: 0 0 18px rgba(174,243,89,0.3);
}

/* ── File uploader ── */
.stFileUploader > div {
    background: var(--panel) !important;
    border: 1px dashed #2a2a30 !important;
    border-radius: 16px !important;
}
.stFileUploader label { color: var(--text-dim) !important; }
.stFileUploader:hover > div { border-color: var(--neon-cyan) !important; }

/* ── Batch table ── */
.btable { width: 100%; border-collapse: collapse; font-size: 0.86rem; margin-top: 0.6rem; }
.btable th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 0.8rem;
    text-align: left;
}
.btable td { padding: 0.6rem 0.8rem; border-bottom: 1px solid #0c0c0c; color: var(--text); }
.btable .pred { color: var(--neon-cyan); font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.btable .none { color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }

/* ── Info tag ── */
.info-tag {
    display: inline-block;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.18rem 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--neon-cyan);
}

/* ── Status pill ── */
.ready-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--neon-lime);
    background: rgba(174,243,89,0.07);
    border: 1px solid rgba(174,243,89,0.3);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    margin: 1rem 0 1.4rem;
}
.ready-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--neon-lime);
    box-shadow: 0 0 8px var(--neon-lime);
}

/* ── Divider ── */
hr { border-color: var(--border); margin: 2.4rem 0; }

/* ── Progress / spinner ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--neon-cyan), var(--neon-violet)) !important; }

/* ── audio player tint ── */
audio { filter: invert(0.06) hue-rotate(180deg); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Acoustic Fingerprint Engine</div>
  <h1 class="hero-title">SOUND<span class="accent">PRINT</span></h1>
  <div class="hero-sub">Index a library as constellation fingerprints, then identify any short clip
  against it — built on spectral peak matching and offset-alignment voting.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME — neon on pure black
# ─────────────────────────────────────────────────────────────────────────────
BG      = '#000000'
AX_BG   = '#0a0a0a'
GRID    = '#1f1f1f'
AMBER   = '#00f5ff'   # primary accent line/fill (was amber, now neon cyan)
ORANGE  = '#ff00e5'   # secondary accent (was orange, now neon pink)
TEXT    = '#6e6e76'
FG      = '#f2f2f7'
TEAL_PT = '#aef359'   # constellation dot color (neon lime)


def _style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)


def dark_fig(w=10, h=3.5, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        _style_ax(ax, fig)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS — UNCHANGED ALGORITHM (do not modify logic below)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading song database…")
def load_database(path):
    db = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song  = row['song_name']
            h     = (int(row['freq1']), int(row['freq2']), int(row['dt']))
            times = list(map(int, row['anchor_times'].split(';')))
            if song not in db:
                db[song] = {}
            db[song][h] = times
    return db


def get_constellation(y):
    D         = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    D_db      = librosa.amplitude_to_db(D, ref=np.max)
    local_max = maximum_filter(D_db, size=NEIGH)
    peak_mask = (D_db == local_max) & (D_db > THRESH)
    freq_idx, time_idx = np.where(peak_mask)
    order     = np.argsort(time_idx)
    return time_idx[order], freq_idx[order], D_db


def build_hashes(time_idx, freq_idx):
    hash_map = {}
    for i in range(len(time_idx)):
        t1, f1 = time_idx[i], freq_idx[i]
        pairs  = 0
        for j in range(i + 1, len(time_idx)):
            t2, f2 = time_idx[j], freq_idx[j]
            dt     = t2 - t1
            if dt > DT_MAX: break
            h = (int(f1), int(f2), int(dt))
            if h not in hash_map: hash_map[h] = []
            hash_map[h].append(int(t1))
            pairs += 1
            if pairs >= FAN_OUT: break
    return hash_map


def match_clip(clip_hashes, database):
    scores = {}
    for song_name, song_hm in database.items():
        oc = defaultdict(int)
        for h, cts in clip_hashes.items():
            if h in song_hm:
                for ct in cts:
                    for dt in song_hm[h]:
                        oc[dt - ct] += 1
        if oc:
            best = max(oc, key=oc.get)
            scores[song_name] = (oc[best], dict(oc), best)
        else:
            scores[song_name] = (0, {}, 0)
    return scores


def identify(audio_bytes):
    y, sr       = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    t_idx, f_idx, D_db = get_constellation(y)
    clip_hashes = build_hashes(t_idx, f_idx)
    scores      = match_clip(clip_hashes, database)
    ranked      = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    return y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS (restyled colors only — same data/logic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(D_db):
    fig, (ax,) = dark_fig(9, 3.2)
    img = librosa.display.specshow(D_db, sr=SR, hop_length=HOP,
                                    x_axis='time', y_axis='hz',
                                    cmap='magma', ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title('Spectrogram — query clip', fontsize=9)
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    plt.tight_layout(pad=0.4)
    return fig


def plot_constellation_clip(D_db, time_idx, freq_idx):
    fig, (ax,) = dark_fig(9, 3.2)
    t_sec = librosa.frames_to_time(time_idx, sr=SR, hop_length=HOP)
    f_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[freq_idx]
    ax.scatter(t_sec, f_hz, color=TEAL_PT, s=5, linewidths=0,
               label=f'{len(t_sec)} peaks', alpha=0.9)
    ax.set_ylim(0, 8000)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(f'Constellation map — {len(t_sec)} most prominent peaks', fontsize=9)
    leg = ax.legend(fontsize=7, loc='upper right')
    leg.get_frame().set_facecolor(AX_BG)
    for t in leg.get_texts(): t.set_color(TEAL_PT)
    plt.tight_layout(pad=0.4)
    return fig


def plot_full_song_constellation(best_song, best_offset, clip_len_frames):
    """
    Step 2: reconstruct the full song fingerprint from the database,
    plot as freq_bin vs time (frames), highlight where the clip sits.
    """
    song_hm    = database[best_song]
    all_times, all_freqs = [], []
    for (f1, f2, dt), anchors in song_hm.items():
        for t in anchors:
            all_times.append(t)
            all_freqs.append(f1)

    all_times  = np.array(all_times)
    all_freqs  = np.array(all_freqs)

    clip_start = best_offset
    clip_end   = best_offset + clip_len_frames

    fig, (ax,) = dark_fig(11, 3.8)
    ax.scatter(all_times, all_freqs, color=TEAL_PT, s=1.5,
               linewidths=0, alpha=0.6)
    ax.axvspan(clip_start, clip_end, color=AMBER, alpha=0.12, zorder=3)
    ax.axvline(clip_start, color=AMBER, linewidth=1.2, linestyle='--', alpha=0.9)
    ax.axvline(clip_end,   color=AMBER, linewidth=1.2, linestyle='--', alpha=0.9)

    ax.set_xlabel('time (frames)')
    ax.set_ylabel('freq bin')
    ax.set_title(f'Full fingerprint of "{best_song}" — highlighted window = your clip', fontsize=9)
    plt.tight_layout(pad=0.4)
    return fig


def plot_alignment_spike(scores, ranked):
    correct_song = ranked[0][0]
    oc           = scores[correct_song][1]
    best_offset  = scores[correct_song][2]
    best_count   = scores[correct_song][0]

    fig, (ax,) = dark_fig(11, 3.8)

    if oc:
        offsets = np.array(list(oc.keys()))
        counts  = np.array(list(oc.values()))
        span    = offsets.max() - offsets.min()

        ax.bar(offsets, counts, width=1, color='#142a3a', alpha=0.8)
        ax.bar([best_offset], [best_count],
               width=max(span * 0.008, 3),
               color=AMBER, zorder=5)

        ax.annotate(
            f'{best_count:,} hashes\nalign here',
            xy=(best_offset, best_count),
            xytext=(best_offset + max(span * 0.05, 50), best_count * 0.75),
            fontsize=8,
            color=AMBER,
            fontfamily='monospace',
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2)
        )
        noise_floor = np.median(counts)
        ax.annotate(
            f'chance matches\n(noise floor)',
            xy=(offsets.max() * 0.85, noise_floor + 1),
            fontsize=7, color=TEXT, fontfamily='monospace'
        )

    ax.set_xlabel('time offset  (database frame − query frame)')
    ax.set_ylabel('# hashes')
    ax.set_title('The alignment spike — genuine match makes hashes converge', fontsize=9)
    plt.tight_layout(pad=0.4)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE SCORE BARS HTML
# ─────────────────────────────────────────────────────────────────────────────

def score_bars(ranked, top_n=5):
    max_sc = ranked[0][1][0] if ranked else 1
    rows   = ""
    for i, (song, (sc, _, __)) in enumerate(ranked[:top_n]):
        pct   = int(sc / max_sc * 100) if max_sc > 0 else 0
        fill  = f'score-fill-gold' if i == 0 else 'score-fill-dim'
        nclass= 'score-num-top' if i == 0 else ''
        rows += f"""
        <div class="score-row">
          <div class="score-name">{song}</div>
          <div class="score-track"><div class="{fill}" style="width:{pct}%"></div></div>
          <div class="score-num {nclass}">{sc:,}</div>
        </div>"""
    return f'<div style="margin:0.5rem 0 1.5rem">{rows}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATABASE
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DB_PATH):
    st.error(f"Database file **'{DB_PATH}'** not found. "
             f"Make sure it's in the same folder as app.py.")
    st.stop()

database = load_database(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_lib, tab_id, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

# ══════════════════════════════════════════════════════════════════════════════
# LIBRARY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown('<div class="sec-eyebrow">INDEXED CATALOG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Library</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="admin-box">
        Song indexing is managed by the admin · <b style="color:#aef359">{len(database)} tracks</b> indexed.<br>
        Drop a clip in the <b style="color:#00f5ff">Identify</b> tab to test the library.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cand-label">IN THE DATABASE</div>', unsafe_allow_html=True)

    songs = sorted(database.keys())
    cols  = st.columns(4)

    for i, song in enumerate(songs):
        n_hashes = len(database[song])
        with cols[i % 4]:
            fig_m, ax_m = plt.subplots(figsize=(2.2, 1.5))
            fig_m.patch.set_facecolor(AX_BG)
            ax_m.set_facecolor(AX_BG)
            ax_m.axis('off')
            rng = np.random.default_rng(seed=abs(hash(song)) % 99999)
            n   = min(n_hashes, 400)
            xs  = rng.uniform(0, 100, n)
            ys  = rng.uniform(0, 512, n)
            neon_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "neon", ["#00f5ff", "#9d4dff", "#ff00e5"]
            )
            c   = neon_cmap(rng.uniform(0.0, 1.0, n))
            ax_m.scatter(xs, ys, s=1.5, c=c, linewidths=0)
            plt.tight_layout(pad=0)
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
            st.markdown(f"""
            <div class="song-card" style="margin-top:-1.4rem; border-top-left-radius:0; border-top-right-radius:0;">
              <div class="card-name">{song}</div>
              <div class="card-meta">{n_hashes:,} HASHES</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_id:
    st.markdown('<div class="sec-eyebrow">CLIP RECOGNITION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Identify a clip</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an audio clip here",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        label_visibility="collapsed"
    )

    if uploaded:
        audio_bytes = uploaded.read()

        st.markdown("""
        <div class="ready-pill"><div class="ready-dot"></div> CLIP LOADED — READY TO SCAN</div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.audio(audio_bytes)
        with col_b:
            scan_clicked = st.button("▶  START SCAN", use_container_width=True)

        # Only run the (expensive) matching pipeline once the user presses start
        if scan_clicked or st.session_state.get("scan_done"):
            st.session_state["scan_done"] = True

            with st.spinner("Fingerprinting and matching…"):
                y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked = identify(audio_bytes)

            best_song   = ranked[0][0]
            best_score  = ranked[0][1][0]
            best_offset = ranked[0][1][2]          # frame offset
            runner_up   = ranked[1][1][0] if len(ranked) > 1 and ranked[1][1][0] > 0 else 1
            ratio       = round(best_score / max(runner_up, 1))
            clip_frames = len(y) // HOP

            st.divider()

            # ── Match card ────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="match-wrap">
              <div class="match-eyebrow">◉ MATCH FOUND</div>
              <div class="match-song">{best_song}</div>
              <div class="match-stats">
                cluster score <b>{best_score:,}</b> &nbsp;·&nbsp;
                <b>{ratio:,}×</b> the runner-up
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Candidate scores ──────────────────────────────────────────────
            st.markdown('<div class="cand-label">CANDIDATE SCORES</div>', unsafe_allow_html=True)
            st.markdown(score_bars(ranked), unsafe_allow_html=True)

            st.divider()

            # ── STEP 1: Spectrogram → Constellation ───────────────────────────
            st.markdown(f"""
            <div class="step-wrap">
              <div class="step-eye">STEP 1 · FEATURE EXTRACTION</div>
              <div class="step-title">From spectrogram to constellation</div>
              <div class="step-desc">
                The clip was converted into a time-frequency map (left); brighter means
                louder at that frequency and moment. From that image, only the
                <b>{len(t_idx)} most prominent peaks</b> were kept (right). Discarding
                amplitude and phase makes the fingerprint robust to EQ, volume changes,
                and mild noise.
              </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_spectrogram(D_db), use_container_width=True)
            with c2:
                st.pyplot(plot_constellation_clip(D_db, t_idx, f_idx), use_container_width=True)

            # ── STEP 2: Full song constellation with clip window ──────────────
            st.markdown(f"""
            <div class="step-wrap">
              <div class="step-eye">STEP 2 · DATABASE SEARCH</div>
              <div class="step-title">Where in the song?</div>
              <div class="step-desc">
                The <b>{len(clip_hashes):,} fingerprint hashes</b> were looked up against
                every indexed track. Below is the full fingerprint of
                <b>{best_song}</b> reconstructed from the database — each dot is a stored
                hash anchor. The highlighted band is exactly where your clip sits
                inside the full song.
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.pyplot(
                plot_full_song_constellation(best_song, best_offset, clip_frames),
                use_container_width=True
            )

            # ── STEP 3: Alignment spike ───────────────────────────────────────
            st.markdown(f"""
            <div class="step-wrap">
              <div class="step-eye">STEP 3 · THE PROOF</div>
              <div class="step-title">The alignment spike</div>
              <div class="step-desc">
                Every matched hash votes for a time offset (database frame minus query
                frame). Chance matches scatter votes randomly, forming a flat noise floor.
                A genuine match makes them converge:
                <b>{best_score:,} hashes agreed on a single offset</b>.
                That spike cannot be a coincidence.
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.pyplot(plot_alignment_spike(scores, ranked), use_container_width=True)
    else:
        st.session_state["scan_done"] = False

# ══════════════════════════════════════════════════════════════════════════════
# BATCH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-eyebrow">BULK PROCESSING</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Identify many clips at once</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-desc" style="margin-bottom:1.8rem">
    Upload a set of query clips. Each is identified against the currently indexed
    library, and the results are written to a standardised
    <span class="info-tag">results.csv</span> with columns
    <span class="info-tag">filename</span>
    <span class="info-tag">prediction</span>.
    The prediction is the matched track's filename without its extension,
    or <span class="info-tag">none</span> when no candidate clears the confidence threshold.
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload clips",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files:
        st.markdown(f"""
        <div class="ready-pill"><div class="ready-dot"></div> {len(batch_files)} CLIPS LOADED — READY TO SCAN</div>
        """, unsafe_allow_html=True)

        if st.button("▶  RUN BATCH SCAN"):
            results  = []
            progress = st.progress(0, text="Processing clips…")

            for i, f in enumerate(batch_files):
                try:
                    _, _, _, _, _, _, sc, rnk = identify(f.read())
                    top_score = rnk[0][1][0]
                    runner    = rnk[1][1][0] if len(rnk) > 1 else 0
                    # Confidence gate: must be at least 10× runner-up
                    pred = rnk[0][0] if (runner == 0 or top_score / max(runner, 1) >= 10) else "none"
                except Exception:
                    pred = "none"

                results.append({
                    "filename":   os.path.splitext(f.name)[0],
                    "prediction": pred
                })
                progress.progress(
                    (i + 1) / len(batch_files),
                    text=f"Processed {i+1}/{len(batch_files)}: {f.name}"
                )

            progress.empty()
            df = pd.DataFrame(results)

            # Results table
            st.markdown('<div class="cand-label" style="margin-top:1.5rem">RESULTS</div>',
                        unsafe_allow_html=True)

            matched = sum(1 for r in results if r["prediction"] != "none")
            rows_html = "".join(
                f'<tr>'
                f'<td>{r["filename"]}</td>'
                f'<td class="{"pred" if r["prediction"] != "none" else "none"}">'
                f'{r["prediction"]}</td>'
                f'</tr>'
                for r in results
            )
            st.markdown(f"""
            <table class="btable">
              <thead><tr><th>FILE</th><th>PREDICTION</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            <div style="color:#6e6e76;font-family:'JetBrains Mono',monospace;
                        font-size:0.7rem;margin-top:0.85rem">
              {matched} / {len(batch_files)} clips matched to a track
              ({len(batch_files)-matched} returned none).
            </div>
            """, unsafe_allow_html=True)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇  DOWNLOAD RESULTS.CSV",
                csv_bytes, "results.csv", "text/csv"
            )