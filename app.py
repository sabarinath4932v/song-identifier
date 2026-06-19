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

# ── Constants (ALGORITHMS UNCHANGED) ──────────────────────────────────────────
DB_PATH = "fingerprint_database_light.csv"
SR      = 22050
N_FFT   = 2048
HOP     = 512
NEIGH   = 20
THRESH  = -25
FAN_OUT = 3
DT_MAX  = 60

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AURA · Neural Audio Scan",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state="expanded"
)

# ── CSS — Pure Black, Apple-Sleek, Neon Accents ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Base Variables & Global ── */
:root {
    --bg: #000000;
    --panel: #090909;
    --border: #1a1a1a;
    --text-main: #f5f5f7;
    --text-muted: #86868b;
    --neon-cyan: #00f3ff;
    --neon-pink: #ff007f;
}

html, body, [class*="css"] {
    background-color: var(--bg);
    color: var(--text-main);
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp { background-color: var(--bg); }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid var(--border);
}

/* ── Typography & Headers ── */
.app-title {
    font-family: 'Inter', sans-serif;
    font-weight: 100;
    font-size: 3.5rem;
    letter-spacing: -0.04em;
    color: var(--text-main);
    margin: 0;
    line-height: 1;
}
.app-title b {
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, var(--neon-cyan), var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 0.9rem;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
    margin-bottom: 3rem;
}

/* ── Cards & Panels ── */
.glass-panel {
    background: rgba(10, 10, 10, 0.6);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.glass-panel:hover {
    border-color: rgba(0, 243, 255, 0.3);
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.05);
}

/* ── Match Result (Neon Glow) ── */
.match-hero {
    background: #000000;
    border: 1px solid var(--neon-cyan);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(0, 243, 255, 0.15);
}
.match-hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,243,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.match-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.match-song {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
    color: #ffffff;
}
.match-stats {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 1rem;
}
.match-stats span { color: var(--neon-pink); font-weight: 700; }

/* ── Buttons ── */
.stButton > button {
    background-color: transparent !important;
    color: var(--neon-cyan) !important;
    border: 1px solid var(--neon-cyan) !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.8rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: var(--neon-cyan) !important;
    color: #000000 !important;
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.4) !important;
}

/* ── File Uploader ── */
.stFileUploader > div {
    background-color: var(--panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: var(--neon-pink) !important;
}

/* ── Candidate Score Bars ── */
.cand-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
}
.cand-name {
    width: 250px;
    font-size: 0.9rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 300;
}
.cand-track {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    position: relative;
}
.cand-fill-cyan { background: var(--neon-cyan); height: 100%; border-radius: 2px; box-shadow: 0 0 10px var(--neon-cyan); }
.cand-fill-muted { background: #333; height: 100%; border-radius: 2px; }
.cand-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    min-width: 50px;
    text-align: right;
}

/* ── Steps ── */
.step-title {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.step-desc {
    font-size: 0.95rem;
    font-weight: 300;
    color: var(--text-muted);
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* ── Grid/Tables ── */
.lib-grid-name { font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; }
.lib-grid-hash { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--neon-pink); }
hr { border-color: var(--border); margin: 3rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS (ALGORITHMS UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing Database...")
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
# PLOT FUNCTIONS (Updated for Pure Black & Neon Aesthetic)
# ─────────────────────────────────────────────────────────────────────────────

PLOT_BG = '#000000'
AX_BG   = '#050505'
GRID_C  = '#1a1a1a'
CYAN    = '#00f3ff'
PINK    = '#ff007f'
MUTED   = '#86868b'

def _style_ax(ax, fig):
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color('#ffffff')
    ax.title.set_fontsize(10)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.8)

def pure_fig(w=10, h=3.5, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    if ncols == 1: axes = [axes]
    for ax in axes: _style_ax(ax, fig)
    return fig, axes

def plot_spectrogram(D_db):
    fig, (ax,) = pure_fig(9, 3.2)
    img = librosa.display.specshow(D_db, sr=SR, hop_length=HOP,
                                    x_axis='time', y_axis='hz',
                                    cmap='magma', ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title('FREQUENCY MAP (SPECTROGRAM)')
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    plt.tight_layout(pad=0.4)
    return fig

def plot_constellation_clip(D_db, time_idx, freq_idx):
    fig, (ax,) = pure_fig(9, 3.2)
    t_sec = librosa.frames_to_time(time_idx, sr=SR, hop_length=HOP)
    f_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[freq_idx]
    ax.scatter(t_sec, f_hz, color=CYAN, s=8, linewidths=0, alpha=0.9)
    ax.set_ylim(0, 8000)
    ax.set_xlabel('TIME (S)')
    ax.set_ylabel('FREQUENCY (HZ)')
    ax.set_title(f'PEAK EXTRACTION ({len(t_sec)} ANCHORS)')
    plt.tight_layout(pad=0.4)
    return fig

def plot_full_song_constellation(best_song, best_offset, clip_len_frames):
    song_hm = database[best_song]
    all_times, all_freqs = [], []
    for (f1, f2, dt), anchors in song_hm.items():
        for t in anchors:
            all_times.append(t)
            all_freqs.append(f1)
            
    fig, (ax,) = pure_fig(11, 3.5)
    ax.scatter(np.array(all_times), np.array(all_freqs), color='#333333', s=2, linewidths=0)
    
    clip_start, clip_end = best_offset, best_offset + clip_len_frames
    ax.axvspan(clip_start, clip_end, color=PINK, alpha=0.15, zorder=3)
    ax.axvline(clip_start, color=PINK, linewidth=1.5, linestyle=':')
    ax.axvline(clip_end,   color=PINK, linewidth=1.5, linestyle=':')

    ax.set_xlabel('TIME (FRAMES)')
    ax.set_ylabel('FREQUENCY BIN')
    ax.set_title(f'DATABASE MATCH WINDOW: {best_song.upper()}')
    plt.tight_layout(pad=0.4)
    return fig

def plot_alignment_spike(scores, ranked):
    correct_song = ranked[0][0]
    oc           = scores[correct_song][1]
    best_offset  = scores[correct_song][2]
    best_count   = scores[correct_song][0]

    fig, (ax,) = pure_fig(11, 3.5)
    if oc:
        offsets = np.array(list(oc.keys()))
        counts  = np.array(list(oc.values()))
        span    = offsets.max() - offsets.min()

        ax.bar(offsets, counts, width=1, color='#111111')
        ax.bar([best_offset], [best_count], width=max(span*0.01, 3), color=CYAN, zorder=5)

        ax.annotate(
            f'SPIKE: {best_count} HASHES',
            xy=(best_offset, best_count),
            xytext=(best_offset + max(span * 0.05, 50), best_count * 0.8),
            fontsize=9, color=CYAN, fontfamily='monospace',
            arrowprops=dict(arrowstyle='-', color=CYAN, lw=1)
        )
    ax.set_xlabel('TIME OFFSET (DB FRAME - QUERY FRAME)')
    ax.set_ylabel('VOTE COUNT')
    ax.set_title('ALIGNMENT VERIFICATION')
    plt.tight_layout(pad=0.4)
    return fig

def render_score_bars(ranked, top_n=5):
    max_sc = ranked[0][1][0] if ranked else 1
    html = ""
    for i, (song, (sc, _, __)) in enumerate(ranked[:top_n]):
        pct = int(sc / max_sc * 100) if max_sc > 0 else 0
        fill_class = 'cand-fill-cyan' if i == 0 else 'cand-fill-muted'
        color_sty  = 'color: var(--neon-cyan);' if i==0 else ''
        html += f"""
        <div class="cand-row">
            <div class="cand-name">{song}</div>
            <div class="cand-track"><div class="{fill_class}" style="width:{pct}%"></div></div>
            <div class="cand-score" style="{color_sty}">{sc:,}</div>
        </div>
        """
    return html

# ─────────────────────────────────────────────────────────────────────────────
# INIT & ROUTING
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at `{DB_PATH}`.")
    st.stop()

database = load_database(DB_PATH)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("<h2 style='font-weight:200; letter-spacing: 0.1em;'>AURA.SYS</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    nav_selection = st.radio("MODULE", ["SINGLE SCAN", "BATCH PROCESS", "DATABASE LIBRARY"], label_visibility="collapsed")
    st.markdown("<div style='margin-top: 50vh; color: #444; font-size: 0.8rem;'>EE200 // v2.0</div>", unsafe_allow_html=True)

# MAIN HEADER
st.markdown("""
<h1 class="app-title">Audio<b>ID</b></h1>
<div class="app-subtitle">NEURAL FINGERPRINTING & RECOGNITION ENGINE</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1: SINGLE SCAN
# ══════════════════════════════════════════════════════════════════════════════
if nav_selection == "SINGLE SCAN":
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader("DROP AUDIO TARGET", type=["mp3", "wav", "flac", "m4a", "ogg"])
    
    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Explicit Button to trigger Scan
        if st.button("INITIATE SCAN"):
            with st.spinner("EXTRACTING AUDIO FEATURES..."):
                y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked = identify(audio_bytes)

            best_song   = ranked[0][0]
            best_score  = ranked[0][1][0]
            best_offset = ranked[0][1][2]
            runner_up   = ranked[1][1][0] if len(ranked) > 1 and ranked[1][1][0] > 0 else 1
            ratio       = round(best_score / max(runner_up, 1))
            clip_frames = len(y) // HOP

            # Match Card
            st.markdown(f"""
            <div class="match-hero" style="margin-top: 2rem;">
                <div class="match-eyebrow">TARGET IDENTIFIED</div>
                <div class="match-song">{best_song}</div>
                <div class="match-stats">
                    CONFIDENCE SCORE: <span>{best_score:,}</span> &nbsp;//&nbsp; 
                    SEPARATION: <span>{ratio:,}X</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Scores
            st.markdown("<div class="step-title" style='font-size:1rem; color:var(--neon-cyan);'>CANDIDATE PROBABILITY</div>", unsafe_allow_html=True)
            st.markdown(render_score_bars(ranked), unsafe_allow_html=True)

            st.divider()

            # Breakdown Analysis
            st.markdown("<div class='step-title'>01. Spectral Topography</div>", unsafe_allow_html=True)