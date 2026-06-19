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

# ── Constants ──────────────────────────────────────────────────────────────────
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
    page_title = "Soundprint · Audio ID",
    page_icon  = "🔊",
    layout     = "wide"
)

# ── CSS — deep navy + amber/gold theme, IBM Plex fonts ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Serif:wght@600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    background-color: #080c14;
    color: #c8cfe0;
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background-color: #080c14; }

/* ── Header ── */
.hdr {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #141c2e;
    margin-bottom: 0;
}
.hdr-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    color: #f5a623;
    margin-bottom: 0.5rem;
}
.hdr-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #f0f4ff;
    margin: 0;
    line-height: 1.1;
}
.hdr-title em { color: #f5a623; font-style: normal; }
.hdr-sub {
    font-size: 0.9rem;
    color: #5a6a8a;
    margin-top: 0.6rem;
    font-weight: 300;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #141c2e;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #3a4a6a;
    background: transparent !important;
    border: none !important;
    padding: 1rem 1.8rem;
}
.stTabs [aria-selected="true"] {
    color: #f5a623 !important;
    border-bottom: 2px solid #f5a623 !important;
}

/* ── Cards ── */
.song-card {
    background: #0d1320;
    border: 1px solid #141c2e;
    border-radius: 6px;
    padding: 0.9rem;
    margin-bottom: 0.6rem;
}
.song-card:hover { border-color: #f5a623; }
.card-name { font-size: 0.78rem; font-weight: 600; color: #c8cfe0; margin: 0.4rem 0 0.2rem; }
.card-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #3a4a6a; }

/* ── Match result ── */
.match-wrap {
    background: linear-gradient(135deg, #0d1320 0%, #111a2e 100%);
    border: 1px solid #f5a623;
    border-radius: 10px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.match-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f5a623, #ff6b35, #f5a623);
}
.match-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #f5a623;
    margin-bottom: 0.6rem;
}
.match-song {
    font-family: 'IBM Plex Serif', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #f0f4ff;
    margin: 0;
    line-height: 1.15;
}
.match-stats {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #5a6a8a;
    margin-top: 0.6rem;
}
.match-stats b { color: #f5a623; }

/* ── Candidate scores ── */
.cand-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #3a4a6a;
    margin: 1.5rem 0 0.75rem;
}
.score-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
.score-name { color: #c8cfe0; min-width: 230px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.score-track { flex: 1; background: #141c2e; border-radius: 2px; height: 5px; }
.score-fill-gold { background: linear-gradient(90deg, #f5a623, #ff6b35); height: 5px; border-radius: 2px; }
.score-fill-dim  { background: #1e2a42; height: 5px; border-radius: 2px; }
.score-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #3a4a6a; min-width: 45px; text-align: right; }
.score-num-top { color: #f5a623; }

/* ── Step headers ── */
.step-wrap {
    border-left: 3px solid #f5a623;
    padding-left: 1.2rem;
    margin: 2.5rem 0 1rem;
}
.step-eye {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    color: #f5a623;
    margin-bottom: 0.3rem;
}
.step-title { font-size: 1.4rem; font-weight: 700; color: #f0f4ff; margin: 0; }
.step-desc { font-size: 0.85rem; color: #5a6a8a; margin-top: 0.4rem; line-height: 1.65; }
.step-desc b { color: #f5a623; }

/* ── Library admin box ── */
.admin-box {
    background: #0d1320;
    border: 1px solid #141c2e;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #3a4a6a;
    margin-bottom: 2rem;
    line-height: 1.9;
}

/* ── Buttons ── */
.stButton > button {
    background: #f5a623;
    color: #080c14;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 5px;
    padding: 0.65rem 1.6rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #ffbe4f; }

/* ── File uploader ── */
.stFileUploader > div {
    background: #0d1320 !important;
    border: 1px dashed #1e2a42 !important;
    border-radius: 8px !important;
}

/* ── Batch table ── */
.btable { width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 0.5rem; }
.btable th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #3a4a6a;
    border-bottom: 1px solid #141c2e;
    padding: 0.5rem 0.75rem;
    text-align: left;
}
.btable td { padding: 0.55rem 0.75rem; border-bottom: 1px solid #0d1320; color: #c8cfe0; }
.btable .pred { color: #f5a623; font-family: 'IBM Plex Mono', monospace; }
.btable .none { color: #3a4a6a; font-family: 'IBM Plex Mono', monospace; }

/* ── Info tag ── */
.info-tag {
    display: inline-block;
    background: #111a2e;
    border: 1px solid #1e2a42;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #f5a623;
}

/* ── Divider ── */
hr { border-color: #141c2e; margin: 2rem 0; }

/* ── Spinner / progress ── */
.stProgress > div > div { background: #f5a623; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">SIGNALS · SYSTEMS · NETWORKS &nbsp;·&nbsp; EE200 PROJECT</div>
  <div class="hdr-title">Sound<em>print</em></div>
  <div class="hdr-sub">Index a library of songs as spectrogram fingerprints — then identify any short clip against it.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────
BG      = '#080c14'
AX_BG   = '#0d1320'
GRID    = '#141c2e'
AMBER   = '#f5a623'
ORANGE  = '#ff6b35'
TEXT    = '#5a6a8a'
FG      = '#c8cfe0'
TEAL_PT = '#4de8c2'


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
# CORE FUNCTIONS
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
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(D_db):
    fig, (ax,) = dark_fig(9, 3.2)
    img = librosa.display.specshow(D_db, sr=SR, hop_length=HOP,
                                    x_axis='time', y_axis='hz',
                                    cmap='inferno', ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title('Spectrogram — query clip', fontsize=9)
    cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    plt.tight_layout(pad=0.4)
    return fig


def plot_constellation_clip(D_db, time_idx, freq_idx):
    fig, (ax,) = dark_fig(9, 3.2)
    # dark background scatter only (no spectrogram underneath — matches prof's style)
    t_sec = librosa.frames_to_time(time_idx, sr=SR, hop_length=HOP)
    f_hz  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)[freq_idx]
    ax.scatter(t_sec, f_hz, color=TEAL_PT, s=5, linewidths=0,
               label=f'{len(t_sec)} peaks', alpha=0.85)
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
    Matches image 4/5 from the prof's demo.
    """
    song_hm    = database[best_song]
    # Collect all anchor points: (time_frame, freq_bin)
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
    # All song dots
    ax.scatter(all_times, all_freqs, color=TEAL_PT, s=1.5,
               linewidths=0, alpha=0.6)
    # Highlight clip window
    ax.axvspan(clip_start, clip_end, color=AMBER, alpha=0.12, zorder=3)
    ax.axvline(clip_start, color=AMBER, linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(clip_end,   color=AMBER, linewidth=1.2, linestyle='--', alpha=0.8)

    ax.set_xlabel('time (frames)')
    ax.set_ylabel('freq bin')
    ax.set_title(f'Full fingerprint of "{best_song}" — highlighted window = your clip', fontsize=9)
    plt.tight_layout(pad=0.4)
    return fig


def plot_alignment_spike(scores, ranked):
    """
    Step 3: offset histogram for correct song — big orange spike.
    Matches image 6 from the prof's demo.
    """
    correct_song = ranked[0][0]
    oc           = scores[correct_song][1]   # dict of offset→count
    best_offset  = scores[correct_song][2]
    best_count   = scores[correct_song][0]

    fig, (ax,) = dark_fig(11, 3.8)

    if oc:
        offsets = np.array(list(oc.keys()))
        counts  = np.array(list(oc.values()))

        # All bars in dim teal
        ax.bar(offsets, counts, width=1, color='#1e4a5a', alpha=0.7)
        # The winning spike in amber/orange
        ax.bar([best_offset], [best_count],
               width=max(offsets.ptp() * 0.008, 3),
               color=AMBER, zorder=5)

        # Annotation
        ax.annotate(
            f'{best_count:,} hashes\nalign here',
            xy=(best_offset, best_count),
            xytext=(best_offset + max(offsets.ptp() * 0.05, 50),
                    best_count * 0.75),
            fontsize=8,
            color=AMBER,
            fontfamily='IBM Plex Mono',
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2)
        )
        # Noise floor annotation
        noise_floor = np.median(counts)
        ax.annotate(
            f'chance matches\n(noise floor)',
            xy=(offsets.max() * 0.85, noise_floor + 1),
            fontsize=7, color=TEXT, fontfamily='IBM Plex Mono'
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

tab_lib, tab_id, tab_batch = st.tabs(["◆  LIBRARY", "◎  IDENTIFY", "▦  BATCH"])

# ══════════════════════════════════════════════════════════════════════════════
# LIBRARY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown('<div class="cand-label" style="margin-top:1.5rem">LIBRARY</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="admin-box">
        Song indexing is managed by the admin.<br>
        Drop a clip in the <b style="color:#f5a623">Identify</b> tab to test the library.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cand-label">IN THE DATABASE</div>', unsafe_allow_html=True)

    songs = sorted(database.keys())
    cols  = st.columns(4)

    for i, song in enumerate(songs):
        n_hashes = len(database[song])
        with cols[i % 4]:
            # Tiny constellation thumbnail
            fig_m, ax_m = plt.subplots(figsize=(2.2, 1.5))
            fig_m.patch.set_facecolor(AX_BG)
            ax_m.set_facecolor(AX_BG)
            ax_m.axis('off')
            rng = np.random.default_rng(seed=abs(hash(song)) % 99999)
            n   = min(n_hashes, 400)
            xs  = rng.uniform(0, 100, n)
            ys  = rng.uniform(0, 512, n)
            c   = plt.cm.plasma(rng.uniform(0.2, 1.0, n))
            ax_m.scatter(xs, ys, s=1.5, c=c, linewidths=0)
            plt.tight_layout(pad=0)
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
            st.markdown(f"""
            <div class="card-name">{song}</div>
            <div class="card-meta">{n_hashes:,} hashes</div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_id:
    st.markdown('<div class="cand-label" style="margin-top:1.5rem">SEARCH</div>',
                unsafe_allow_html=True)
    st.markdown('<h2 style="color:#f0f4ff;margin:0 0 1rem">Identify a clip</h2>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an audio clip here",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        label_visibility="collapsed"
    )

    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)

        with st.spinner("Fingerprinting and matching…"):
            y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked = identify(audio_bytes)

        best_song   = ranked[0][0]
        best_score  = ranked[0][1][0]
        best_offset = ranked[0][1][2]          # frame offset
        runner_up   = ranked[1][1][0] if len(ranked) > 1 and ranked[1][1][0] > 0 else 1
        ratio       = round(best_score / max(runner_up, 1))
        clip_frames = len(y) // HOP

        # ── Match card ────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="match-wrap">
          <div class="match-eyebrow">MATCH FOUND</div>
          <div class="match-song">{best_song}</div>
          <div class="match-stats">
            cluster score <b>{best_score:,}</b> &nbsp;·&nbsp;
            <b>{ratio:,}×</b> the runner-up
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Candidate scores ──────────────────────────────────────────────────
        st.markdown('<div class="cand-label">CANDIDATE SCORES</div>', unsafe_allow_html=True)
        st.markdown(score_bars(ranked), unsafe_allow_html=True)

        st.divider()

        # ── STEP 1: Spectrogram → Constellation ───────────────────────────────
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

        # ── STEP 2: Full song constellation with clip window ──────────────────
        st.markdown(f"""
        <div class="step-wrap">
          <div class="step-eye">STEP 2 · DATABASE SEARCH</div>
          <div class="step-title">Where in the song?</div>
          <div class="step-desc">
            The <b>{len(clip_hashes):,} fingerprint hashes</b> were looked up against
            every indexed track. Below is the full fingerprint of
            <b>{best_song}</b> reconstructed from the database — each dot is a stored
            hash anchor. The highlighted amber window is exactly where your clip sits
            inside the full song.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.pyplot(
            plot_full_song_constellation(best_song, best_offset, clip_frames),
            use_container_width=True
        )

        # ── STEP 3: Alignment spike ───────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# BATCH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="cand-label" style="margin-top:1.5rem">BATCH</div>',
                unsafe_allow_html=True)
    st.markdown('<h2 style="color:#f0f4ff;margin:0 0 0.5rem">Identify many clips at once</h2>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="step-desc" style="margin-bottom:1.5rem">
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
        if st.button("▶  Run batch"):
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
            <div style="color:#3a4a6a;font-family:IBM Plex Mono,monospace;
                        font-size:0.7rem;margin-top:0.75rem">
              {matched} / {len(batch_files)} clips matched to a track
              ({len(batch_files)-matched} returned none).
            </div>
            """, unsafe_allow_html=True)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇  Download results.csv",
                csv_bytes, "results.csv", "text/csv"
            )