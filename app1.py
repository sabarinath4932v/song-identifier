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

# ── CSS — Split-panel design: dark sidebar + clean light main ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --sidebar-bg:   #0e0e12;
    --sidebar-w:    260px;
    --main-bg:      #f5f4f0;
    --card-bg:      #ffffff;
    --accent:       #7c5cfc;
    --accent-2:     #c5f135;
    --accent-3:     #fc5c7d;
    --text-dark:    #0e0e12;
    --text-mid:     #666672;
    --text-light:   #b0b0bb;
    --border:       #e8e7e2;
    --sidebar-text: #f0f0f5;
    --sidebar-dim:  #6e6e82;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Full app background ── */
.stApp {
    background: var(--main-bg) !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ── Hide default streamlit chrome ── */
header[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
footer { display: none !important; }

/* ── SIDEBAR — dark panel left ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid #1a1a24 !important;
    width: var(--sidebar-w) !important;
    min-width: var(--sidebar-w) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
    background: var(--sidebar-bg) !important;
}
.stSidebar .block-container { padding: 0 !important; }

/* ── TABS — pill style on light bg ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
    gap: 0.25rem;
    padding: 0.35rem;
    width: auto;
    margin: 0 0 2rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--text-mid) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.6rem;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--sidebar-bg) !important;
    background: var(--accent-2) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(197,241,53,0.3);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { background: transparent !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    border: none;
    border-radius: 10px;
    padding: 0.8rem 2rem;
    transition: all 0.18s ease;
    box-shadow: 0 4px 16px rgba(124,92,252,0.3);
}
.stButton > button:hover {
    background: #6a4aed;
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(124,92,252,0.45);
}
.stButton > button:active { transform: translateY(0); }

.stDownloadButton > button {
    background: transparent;
    color: var(--accent);
    border: 1.5px solid var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    transition: all 0.18s;
}
.stDownloadButton > button:hover {
    background: rgba(124,92,252,0.06);
}

/* ── File uploader ── */
.stFileUploader > div {
    background: var(--card-bg) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}
.stFileUploader label { color: var(--text-mid) !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    border-radius: 4px;
}

/* ── Audio widget ── */
audio { border-radius: 8px; width: 100%; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — dark panel with logo + nav info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 2rem 1.6rem 1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.3em;
                    color:#6e6e82; text-transform:uppercase; margin-bottom:0.6rem;">
            Acoustic Engine
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
                    color:#f0f0f5; letter-spacing:-0.03em; line-height:1;">
            SOUND<span style="color:#c5f135">PRINT</span>
        </div>
        <div style="width:32px; height:3px; background:#c5f135; margin-top:1rem;
                    border-radius:2px;"></div>
    </div>

    <div style="padding: 0 1rem; margin-top:0.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem;
                    letter-spacing:0.25em; color:#3a3a4e; text-transform:uppercase;
                    padding: 0 0.6rem; margin-bottom:0.6rem;">
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # nav items — decorative only, tabs handle routing
    nav_items = [
        ("◈", "Library", "Browse indexed catalog"),
        ("◎", "Identify", "Match a clip"),
        ("⊞", "Batch", "Bulk processing"),
    ]
    for icon, label, sub in nav_items:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.9rem;
                    padding:0.75rem 1.6rem; margin:0.15rem 0.6rem;
                    border-radius:10px; cursor:pointer;
                    transition:background 0.15s;">
            <span style="font-size:1rem; color:#c5f135; width:18px; flex-shrink:0;">{icon}</span>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:0.85rem;
                            font-weight:600; color:#f0f0f5;">{label}</div>
                <div style="font-size:0.67rem; color:#6e6e82; margin-top:0.05rem;">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:#1a1a24; margin:1.2rem 1.6rem;'></div>",
                unsafe_allow_html=True)

    # Algorithm constants display
    st.markdown("""
    <div style="padding:0 1.6rem 1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.58rem;
                    letter-spacing:0.25em; color:#3a3a4e; text-transform:uppercase;
                    margin-bottom:0.8rem;">
            Engine Config
        </div>
    """, unsafe_allow_html=True)

    params = [
        ("SR", "22 050 Hz"),
        ("N_FFT", "2 048"),
        ("HOP", "512"),
        ("NEIGH", "20"),
        ("FAN_OUT", "3"),
        ("DT_MAX", "60 fr"),
    ]
    for k, v in params:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.35rem 0; border-bottom:1px solid #1a1a24;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                         color:#6e6e82;">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                         color:#c5f135;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # bottom badge
    st.markdown("""
    <div style="position:absolute; bottom:1.5rem; left:1.6rem; right:1.6rem;">
        <div style="background:#1a1a24; border-radius:10px; padding:0.8rem 1rem;
                    display:flex; align-items:center; gap:0.7rem;">
            <div style="width:8px; height:8px; border-radius:50%; background:#c5f135;
                        box-shadow:0 0 8px #c5f135; flex-shrink:0;"></div>
            <div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                            color:#c5f135; letter-spacing:0.05em;">SYSTEM READY</div>
                <div style="font-size:0.62rem; color:#3a3a4e; margin-top:0.1rem;">
                    Fingerprint engine online
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME — clean minimal on near-white
# ─────────────────────────────────────────────────────────────────────────────
BG      = '#f5f4f0'
AX_BG   = '#ffffff'
GRID    = '#ebebeb'
ACCENT  = '#7c5cfc'
ACCENT2 = '#c5f135'
ACCENT3 = '#fc5c7d'
TEXT    = '#b0b0bb'
FG      = '#0e0e12'
DOT_CLR = '#7c5cfc'


def _style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)


def dark_fig(w=10, h=3.5, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        _style_ax(ax, fig)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS — UNCHANGED ALGORITHM
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
# PLOT FUNCTIONS (restyled — same logic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(D_db):
    fig, (ax,) = dark_fig(9, 3.2)
    img = librosa.display.specshow(D_db, sr=SR, hop_length=HOP,
                                    x_axis='time', y_axis='hz',
                                    cmap='Blues', ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title('Spectrogram — query clip', fontsize=9, fontweight='600', color=FG)
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    plt.tight_layout(pad=0.6)
    return fig


def plot_constellation_clip(D_db, time_idx, freq_idx):
    fig, (ax,) = dark_fig(9, 3.2)
    t_sec = librosa.frames_to_time(time_idx, sr=SR, hop_length=HOP)
    f_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[freq_idx]
    ax.scatter(t_sec, f_hz, color=DOT_CLR, s=6, linewidths=0,
               label=f'{len(t_sec)} peaks', alpha=0.75)
    ax.set_ylim(0, 8000)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(f'Constellation — {len(t_sec)} most prominent peaks', fontsize=9,
                 fontweight='600', color=FG)
    leg = ax.legend(fontsize=7, loc='upper right')
    leg.get_frame().set_facecolor(AX_BG)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts(): t.set_color(DOT_CLR)
    plt.tight_layout(pad=0.6)
    return fig


def plot_full_song_constellation(best_song, best_offset, clip_len_frames):
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
    ax.scatter(all_times, all_freqs, color='#c0b8f8', s=2,
               linewidths=0, alpha=0.5)
    ax.axvspan(clip_start, clip_end, color=ACCENT, alpha=0.10, zorder=3)
    ax.axvline(clip_start, color=ACCENT, linewidth=1.4, linestyle='--', alpha=0.8)
    ax.axvline(clip_end,   color=ACCENT, linewidth=1.4, linestyle='--', alpha=0.8)

    ax.set_xlabel('time (frames)')
    ax.set_ylabel('freq bin')
    ax.set_title(f'Full fingerprint · "{best_song}" — shaded = your clip',
                 fontsize=9, fontweight='600', color=FG)
    plt.tight_layout(pad=0.6)
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

        ax.bar(offsets, counts, width=1, color='#e8e7f8', alpha=0.9)
        ax.bar([best_offset], [best_count],
               width=max(span * 0.008, 3),
               color=ACCENT, zorder=5)

        ax.annotate(
            f'{best_count:,} hashes\nalign here',
            xy=(best_offset, best_count),
            xytext=(best_offset + max(span * 0.05, 50), best_count * 0.75),
            fontsize=8,
            color=ACCENT,
            fontfamily='monospace',
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2)
        )
        noise_floor = np.median(counts)
        ax.annotate(
            f'noise floor',
            xy=(offsets.max() * 0.85, noise_floor + 1),
            fontsize=7, color=TEXT, fontfamily='monospace'
        )

    ax.set_xlabel('time offset (database frame − query frame)')
    ax.set_ylabel('# matching hashes')
    ax.set_title('Alignment spike — genuine matches converge at a single offset',
                 fontsize=9, fontweight='600', color=FG)
    plt.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_bars_html(ranked, top_n=5):
    max_sc = ranked[0][1][0] if ranked else 1
    rows   = ""
    for i, (song, (sc, _, __)) in enumerate(ranked[:top_n]):
        pct   = int(sc / max_sc * 100) if max_sc > 0 else 0
        is_top = i == 0
        bar_color = f"background:{'linear-gradient(90deg,#7c5cfc,#c5f135)' if is_top else '#e8e7e2'}"
        name_color = "#0e0e12" if is_top else "#b0b0bb"
        num_color  = "#7c5cfc" if is_top else "#b0b0bb"
        rows += f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;">
            <div style="width:18px;height:18px;border-radius:50%;
                        background:{'#7c5cfc' if is_top else '#f0f0f0'};
                        display:flex;align-items:center;justify-content:center;
                        font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        color:{'#fff' if is_top else '#b0b0bb'};
                        flex-shrink:0;">{i+1}</div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:0.82rem;font-weight:{'600' if is_top else '400'};
                            color:{name_color};white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;margin-bottom:0.3rem;">{song}</div>
                <div style="height:5px;background:#f0f0f0;border-radius:999px;overflow:hidden;">
                    <div style="width:{pct}%;height:5px;border-radius:999px;{bar_color};
                                transition:width 0.4s ease;"></div>
                </div>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                        color:{num_color};min-width:52px;text-align:right;
                        font-weight:{'600' if is_top else '400'};">{sc:,}</div>
        </div>"""
    return rows


def stat_card(value, label, accent_color="#7c5cfc", suffix=""):
    return f"""
    <div style="background:#fff;border:1px solid #e8e7e2;border-radius:14px;
                padding:1.4rem 1.6rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
                    background:{accent_color};"></div>
        <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                    color:#0e0e12;letter-spacing:-0.03em;line-height:1;">
            {value}<span style="font-size:1.2rem;color:{accent_color};">{suffix}</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                    letter-spacing:0.18em;color:#b0b0bb;text-transform:uppercase;
                    margin-top:0.4rem;">{label}</div>
    </div>"""


def section_header(eyebrow, title, desc=""):
    return f"""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.3em;color:#7c5cfc;text-transform:uppercase;
                    margin-bottom:0.4rem;">{eyebrow}</div>
        <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:700;
                    color:#0e0e12;letter-spacing:-0.025em;line-height:1.1;">{title}</div>
        {'<div style="font-size:0.88rem;color:#666672;margin-top:0.6rem;max-width:580px;line-height:1.65;">'+desc+'</div>' if desc else ''}
    </div>"""


def step_header(step_num, eyebrow, title, desc):
    return f"""
    <div style="display:flex;gap:1.4rem;align-items:flex-start;margin:2.5rem 0 1.4rem;">
        <div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;
                    background:#7c5cfc;display:flex;align-items:center;
                    justify-content:center;
                    font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                    font-weight:600;color:#fff;margin-top:0.2rem;">
            {step_num:02d}
        </div>
        <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.25em;color:#b0b0bb;text-transform:uppercase;
                        margin-bottom:0.2rem;">{eyebrow}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;
                        color:#0e0e12;letter-spacing:-0.015em;">{title}</div>
            <div style="font-size:0.86rem;color:#666672;margin-top:0.4rem;
                        line-height:1.65;max-width:560px;">{desc}</div>
        </div>
    </div>"""


def divider():
    return '<div style="height:1px;background:#e8e7e2;margin:2rem 0;"></div>'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATABASE
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DB_PATH):
    st.error(f"Database file **'{DB_PATH}'** not found. "
             f"Make sure it's in the same folder as app.py.")
    st.stop()

database = load_database(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — page header + tabs
# ─────────────────────────────────────────────────────────────────────────────

# Page top bar
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:1.6rem 2.4rem 0;margin-bottom:1.8rem;
            border-bottom:1px solid #e8e7e2;padding-bottom:1.2rem;">
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
                    color:#0e0e12;letter-spacing:-0.02em;">
            Acoustic Fingerprint Engine
        </div>
        <div style="font-size:0.8rem;color:#b0b0bb;margin-top:0.15rem;">
            Spectral peak matching &amp; offset-alignment voting
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:1.5rem;">
        <div style="text-align:right;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                        font-weight:600;color:#0e0e12;">{len(database)}</div>
            <div style="font-size:0.68rem;color:#b0b0bb;letter-spacing:0.05em;">
                TRACKS INDEXED
            </div>
        </div>
        <div style="width:1px;height:36px;background:#e8e7e2;"></div>
        <div style="width:10px;height:10px;border-radius:50%;background:#c5f135;
                    box-shadow:0 0 10px #c5f135;"></div>
    </div>
</div>
<div style="padding:0 2.4rem;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_lib, tab_id, tab_batch = st.tabs(["◈  LIBRARY", "◎  IDENTIFY", "⊞  BATCH"])

# ══════════════════════════════════════════════════════════════════════════════
# LIBRARY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown(section_header(
        "INDEXED CATALOG",
        "Library",
        "All tracks fingerprinted and stored in the database. Each constellation visualises the hash distribution of that recording."
    ), unsafe_allow_html=True)

    # Stat row
    total_hashes = sum(len(v) for v in database.values())
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_card(len(database), "Tracks indexed", "#7c5cfc"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card(f"{total_hashes:,}", "Total hashes", "#c5f135"), unsafe_allow_html=True)
    with c3:
        avg = total_hashes // max(len(database), 1)
        st.markdown(stat_card(f"{avg:,}", "Avg hashes/track", "#fc5c7d"), unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.25em;
                color:#b0b0bb;text-transform:uppercase;margin-bottom:1.2rem;
                border-bottom:1px solid #e8e7e2;padding-bottom:0.6rem;">
        Track Catalog
    </div>
    """, unsafe_allow_html=True)

    songs = sorted(database.keys())
    cols  = st.columns(4)

    for i, song in enumerate(songs):
        n_hashes = len(database[song])
        with cols[i % 4]:
            # mini constellation thumbnail
            fig_m, ax_m = plt.subplots(figsize=(2.4, 1.5))
            fig_m.patch.set_facecolor('#f5f4f0')
            ax_m.set_facecolor('#ffffff')
            ax_m.axis('off')
            rng = np.random.default_rng(seed=abs(hash(song)) % 99999)
            n   = min(n_hashes, 400)
            xs  = rng.uniform(0, 100, n)
            ys  = rng.uniform(0, 512, n)
            colors_arr = ['#7c5cfc', '#c5f135', '#fc5c7d', '#c0b8f8']
            c_arr = [colors_arr[int(y/130) % 4] for y in ys]
            ax_m.scatter(xs, ys, s=2, c=c_arr, linewidths=0, alpha=0.7)
            plt.tight_layout(pad=0)
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)

            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e8e7e2;border-top:none;
                        border-bottom-left-radius:12px;border-bottom-right-radius:12px;
                        padding:0.75rem 0.9rem;margin-top:-6px;margin-bottom:1rem;">
                <div style="font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:600;
                            color:#0e0e12;white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;">{song}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                            color:#7c5cfc;margin-top:0.2rem;letter-spacing:0.05em;">
                    {n_hashes:,} hashes
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_id:
    st.markdown(section_header(
        "CLIP RECOGNITION",
        "Identify a Clip",
        "Upload any short audio segment and the engine will fingerprint it, search the database, and show you exactly how the match was found."
    ), unsafe_allow_html=True)

    # Upload area
    uploaded = st.file_uploader(
        "Drop an audio clip here · MP3, WAV, FLAC, M4A, OGG",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        label_visibility="collapsed"
    )

    if uploaded:
        audio_bytes = uploaded.read()

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin:1rem 0 1.4rem;
                    padding:0.6rem 1rem;background:#fff;border:1px solid #e8e7e2;
                    border-radius:10px;width:fit-content;">
            <div style="width:7px;height:7px;border-radius:50%;background:#c5f135;
                        box-shadow:0 0 8px #c5f135;"></div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                         color:#0e0e12;letter-spacing:0.06em;">
                {uploaded.name} — LOADED
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.audio(audio_bytes)
        with col_b:
            scan_clicked = st.button("▶  Scan Clip", use_container_width=True)

        if scan_clicked or st.session_state.get("scan_done"):
            st.session_state["scan_done"] = True

            with st.spinner("Fingerprinting and matching…"):
                y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked = identify(audio_bytes)

            best_song   = ranked[0][0]
            best_score  = ranked[0][1][0]
            best_offset = ranked[0][1][2]
            runner_up   = ranked[1][1][0] if len(ranked) > 1 and ranked[1][1][0] > 0 else 1
            ratio       = round(best_score / max(runner_up, 1))
            clip_frames = len(y) // HOP

            st.markdown(divider(), unsafe_allow_html=True)

            # ── Result: split-panel card (inspired by real-estate split layout) ──
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;
                        border:1px solid #e8e7e2;border-radius:18px;overflow:hidden;
                        margin-bottom:2rem;">
                <!-- left dark panel -->
                <div style="background:#0e0e12;padding:2.4rem 2.6rem;position:relative;
                            overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;right:0;height:3px;
                                background:linear-gradient(90deg,#7c5cfc,#c5f135);"></div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                                letter-spacing:0.3em;color:#c5f135;margin-bottom:0.8rem;
                                text-transform:uppercase;">◉ Match Found</div>
                    <div style="font-family:'Syne',sans-serif;font-size:2.6rem;
                                font-weight:800;color:#f0f0f5;letter-spacing:-0.03em;
                                line-height:1.1;margin-bottom:1.2rem;">
                        {best_song}
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                                color:#6e6e82;line-height:1.9;">
                        Cluster score &nbsp;<span style="color:#c5f135">{best_score:,}</span><br>
                        Runner-up margin &nbsp;<span style="color:#c5f135">{ratio:,}×</span>
                    </div>
                </div>
                <!-- right scores panel -->
                <div style="background:#fff;padding:2.4rem 2.6rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                                letter-spacing:0.25em;color:#b0b0bb;text-transform:uppercase;
                                margin-bottom:1.2rem;">Candidate Scores</div>
                    {score_bars_html(ranked)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Step 1 ────────────────────────────────────────────────────────
            st.markdown(step_header(
                1,
                "Feature Extraction",
                "Spectrogram → Constellation",
                f"The clip's time-frequency map (left) was reduced to its {len(t_idx)} most prominent peaks (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume, and mild noise."
            ), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_spectrogram(D_db), use_container_width=True)
            with c2:
                st.pyplot(plot_constellation_clip(D_db, t_idx, f_idx), use_container_width=True)

            st.markdown(divider(), unsafe_allow_html=True)

            # ── Step 2 ────────────────────────────────────────────────────────
            st.markdown(step_header(
                2,
                "Database Search",
                "Where in the Song?",
                f"The {len(clip_hashes):,} fingerprint hashes were looked up across every indexed track. Below is the full fingerprint of \"{best_song}\" — the shaded band marks exactly where your clip sits."
            ), unsafe_allow_html=True)

            st.pyplot(
                plot_full_song_constellation(best_song, best_offset, clip_frames),
                use_container_width=True
            )

            st.markdown(divider(), unsafe_allow_html=True)

            # ── Step 3 ────────────────────────────────────────────────────────
            st.markdown(step_header(
                3,
                "Proof of Match",
                "The Alignment Spike",
                f"Every matched hash votes for a time offset. Chance matches scatter randomly (flat noise floor). A genuine match makes them converge: {best_score:,} hashes agreed on a single offset. That spike cannot be coincidence."
            ), unsafe_allow_html=True)

            st.pyplot(plot_alignment_spike(scores, ranked), use_container_width=True)

    else:
        st.session_state["scan_done"] = False
        # Upload placeholder card
        st.markdown("""
        <div style="background:#fff;border:1.5px dashed #e8e7e2;border-radius:16px;
                    padding:3.5rem;text-align:center;margin-top:1rem;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">◎</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;
                        color:#0e0e12;margin-bottom:0.4rem;">No clip loaded</div>
            <div style="font-size:0.84rem;color:#b0b0bb;">
                Use the uploader above to drop an audio clip and run identification.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BATCH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(section_header(
        "Bulk Processing",
        "Batch Identify",
        "Upload multiple audio clips at once. Each is matched against the indexed library and results are exported as a structured CSV."
    ), unsafe_allow_html=True)

    # Info row
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;">
        <div style="background:#fff;border:1px solid #e8e7e2;border-radius:12px;
                    padding:1.2rem 1.4rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.2em;color:#b0b0bb;text-transform:uppercase;
                        margin-bottom:0.4rem;">Output format</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                        color:#7c5cfc;">results.csv</div>
        </div>
        <div style="background:#fff;border:1px solid #e8e7e2;border-radius:12px;
                    padding:1.2rem 1.4rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.2em;color:#b0b0bb;text-transform:uppercase;
                        margin-bottom:0.4rem;">Columns</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                        color:#7c5cfc;">filename · prediction</div>
        </div>
        <div style="background:#fff;border:1px solid #e8e7e2;border-radius:12px;
                    padding:1.2rem 1.4rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.2em;color:#b0b0bb;text-transform:uppercase;
                        margin-bottom:0.4rem;">Confidence gate</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                        color:#7c5cfc;">10× runner-up</div>
        </div>
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
        <div style="display:flex;align-items:center;gap:0.6rem;margin:0.8rem 0 1.4rem;
                    padding:0.6rem 1rem;background:#fff;border:1px solid #e8e7e2;
                    border-radius:10px;width:fit-content;">
            <div style="width:7px;height:7px;border-radius:50%;background:#c5f135;
                        box-shadow:0 0 8px #c5f135;"></div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                         color:#0e0e12;letter-spacing:0.06em;">
                {len(batch_files)} CLIPS LOADED
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶  Run Batch Scan"):
            results  = []
            progress = st.progress(0, text="Processing clips…")

            for i, f in enumerate(batch_files):
                try:
                    _, _, _, _, _, _, sc, rnk = identify(f.read())
                    top_score = rnk[0][1][0]
                    runner    = rnk[1][1][0] if len(rnk) > 1 else 0
                    pred = rnk[0][0] if (runner == 0 or top_score / max(runner, 1) >= 10) else "none"
                except Exception:
                    pred = "none"

                results.append({
                    "filename":   os.path.splitext(f.name)[0],
                    "prediction": pred
                })
                progress.progress(
                    (i + 1) / len(batch_files),
                    text=f"Processed {i+1}/{len(batch_files)} — {f.name}"
                )

            progress.empty()
            df = pd.DataFrame(results)

            matched = sum(1 for r in results if r["prediction"] != "none")

            # Summary stat cards
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(stat_card(matched, "Matched", "#c5f135", f"/{len(batch_files)}"), unsafe_allow_html=True)
            with mc2:
                st.markdown(stat_card(len(batch_files)-matched, "No match", "#fc5c7d"), unsafe_allow_html=True)

            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.25em;
                        color:#b0b0bb;text-transform:uppercase;margin:1.5rem 0 0.8rem;
                        border-bottom:1px solid #e8e7e2;padding-bottom:0.6rem;">
                Results
            </div>
            """, unsafe_allow_html=True)

            rows_html = "".join(
                f"""<tr style="border-bottom:1px solid #f0f0f0;">
                    <td style="padding:0.65rem 0.8rem;font-size:0.83rem;color:#0e0e12;">{r["filename"]}</td>
                    <td style="padding:0.65rem 0.8rem;">
                        {"<span style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;font-weight:600;color:#7c5cfc;'>"+r['prediction']+"</span>"
                         if r['prediction'] != 'none'
                         else "<span style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#b0b0bb;'>none</span>"}
                    </td>
                </tr>"""
                for r in results
            )
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e8e7e2;border-radius:14px;overflow:hidden;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid #e8e7e2;background:#fafaf8;">
                            <th style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                                       letter-spacing:0.2em;color:#b0b0bb;padding:0.8rem 0.8rem;
                                       text-align:left;font-weight:500;text-transform:uppercase;">
                                File
                            </th>
                            <th style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                                       letter-spacing:0.2em;color:#b0b0bb;padding:0.8rem 0.8rem;
                                       text-align:left;font-weight:500;text-transform:uppercase;">
                                Prediction
                            </th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.download_button(
                "⬇  Download results.csv",
                csv_bytes, "results.csv", "text/csv"
            )

st.markdown("</div>", unsafe_allow_html=True)