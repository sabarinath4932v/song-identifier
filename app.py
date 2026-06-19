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
import tempfile
from collections import defaultdict
from scipy.ndimage import maximum_filter

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH = "fingerprint_database_light.csv"
SR      = 22050
N_FFT   = 2048
HOP     = 512
NEIGH   = 20
THRESH  = -30
FAN_OUT = 3
DT_MAX  = 80

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "EE200 · Audio Fingerprinting",
    page_icon  = "🎵",
    layout     = "wide"
)

# ── Custom CSS — dark terminal aesthetic ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #0a0a0f; }

/* Header */
.main-header {
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.main-title span { color: #00d4a8; }
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555577;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
}
.tagline { color: #888899; font-size: 0.9rem; margin-top: 0.5rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #555577;
    background: transparent;
    border: none;
    padding: 0.75rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: #00d4a8 !important;
    border-bottom: 2px solid #00d4a8 !important;
    background: transparent !important;
}

/* Cards */
.song-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.song-card:hover { border-color: #00d4a8; }
.card-title { font-size: 0.85rem; font-weight: 600; color: #e0e0e0; margin: 0.5rem 0 0.2rem 0; }
.card-sub { font-size: 0.75rem; color: #555577; font-family: 'Space Mono', monospace; }

/* Match result */
.match-box {
    background: #0f0f1a;
    border: 1px solid #00d4a8;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.match-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #00d4a8;
    margin-bottom: 0.5rem;
}
.match-title { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0; }
.match-score { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #888899; margin-top: 0.5rem; }
.match-score span { color: #00d4a8; font-weight: 700; }

/* Step headers */
.step-box {
    border-left: 3px solid #00d4a8;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
}
.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #00d4a8;
    margin-bottom: 0.25rem;
}
.step-title { font-size: 1.3rem; font-weight: 700; color: #ffffff; margin: 0; }
.step-desc { font-size: 0.85rem; color: #888899; margin-top: 0.4rem; line-height: 1.6; }
.step-desc span { color: #00d4a8; font-weight: 600; }

/* Score bar */
.score-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.score-name { color: #ccccdd; min-width: 220px; }
.score-bar-wrap { flex: 1; background: #1a1a2e; border-radius: 3px; height: 6px; }
.score-bar { background: #00d4a8; height: 6px; border-radius: 3px; }
.score-val { color: #00d4a8; font-family: 'Space Mono', monospace; font-size: 0.75rem; min-width: 45px; text-align: right; }

/* Upload area */
.stFileUploader { background: #0f0f1a; border: 1px dashed #2a2a3e; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: #00d4a8;
    color: #0a0a0f;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.4rem;
}
.stButton > button:hover { background: #00ffcc; }

/* Metric */
.stMetric { background: #0f0f1a; border: 1px solid #1e1e2e; border-radius: 8px; padding: 1rem; }

/* Divider */
hr { border-color: #1e1e2e; }

/* Table */
.results-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.results-table th {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #555577;
    border-bottom: 1px solid #1e1e2e;
    padding: 0.5rem 0.75rem;
    text-align: left;
}
.results-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #0f0f1a; color: #ccccdd; }
.results-table td.pred { color: #00d4a8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="subtitle">SIGNALS, SYSTEMS & NETWORKS · PROJECT DEMO</div>
  <div class="main-title">EE<span>200</span>: Audio Fingerprinting</div>
  <div class="tagline">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
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


def get_constellation(y, sr):
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
    for song_name, song_hash_map in database.items():
        offset_counts = defaultdict(int)
        for h, clip_times in clip_hashes.items():
            if h in song_hash_map:
                for ct in clip_times:
                    for dt in song_hash_map[h]:
                        offset_counts[dt - ct] += 1
        if offset_counts:
            best = max(offset_counts, key=offset_counts.get)
            scores[song_name] = (offset_counts[best], dict(offset_counts))
        else:
            scores[song_name] = (0, {})
    return scores


def identify(audio_bytes):
    y, sr     = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    t_idx, f_idx, D_db = get_constellation(y, sr)
    clip_hashes = build_hashes(t_idx, f_idx)
    scores      = match_clip(clip_hashes, database)
    ranked      = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    return y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS  (dark theme)
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG  = '#0a0a0f'
DARK_AX  = '#0f0f1a'
GRID_COL = '#1e1e2e'
TEAL     = '#00d4a8'
TEXT_COL = '#888899'


def dark_fig(w=10, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values(): sp.set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color('#ccccdd')
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)
    return fig, ax


def plot_spectrogram(D_db, sr):
    fig, ax = dark_fig(10, 3.2)
    img = librosa.display.specshow(D_db, sr=sr, hop_length=HOP,
                                    x_axis='time', y_axis='hz',
                                    cmap='magma', ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title('Spectrogram — query clip', color='#ccccdd', fontsize=10)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    plt.tight_layout()
    return fig


def plot_constellation(D_db, time_idx, freq_idx, sr):
    fig, ax = dark_fig(10, 3.2)
    librosa.display.specshow(D_db, sr=sr, hop_length=HOP,
                              x_axis='time', y_axis='hz',
                              cmap='magma', ax=ax)
    t_sec = librosa.frames_to_time(time_idx, sr=sr, hop_length=HOP)
    f_hz  = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[freq_idx]
    ax.scatter(t_sec, f_hz, color=TEAL, s=6, linewidths=0,
               label=f'{len(t_sec)} peaks', zorder=5)
    ax.set_ylim(0, 8000)
    ax.set_title('Constellation map — standout peaks', color='#ccccdd', fontsize=10)
    leg = ax.legend(fontsize=8, loc='upper right')
    leg.get_frame().set_facecolor(DARK_AX)
    for t in leg.get_texts(): t.set_color(TEAL)
    plt.tight_layout()
    return fig


def plot_alignment_spike(scores, ranked, sr):
    correct_song = ranked[0][0]
    wrong_song   = ranked[1][0] if len(ranked) > 1 else None

    fig = plt.figure(figsize=(12, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for idx, (ax_idx, song) in enumerate([(0, correct_song), (1, wrong_song)]):
        ax = fig.add_subplot(gs[ax_idx])
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color(GRID_COL)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

        if song is None:
            continue

        oc = scores[song][1]
        if not oc:
            ax.set_title(f"{song}\n(no offsets)", color='#ccccdd', fontsize=9)
            continue

        best_offset = max(oc, key=oc.get)
        best_count  = oc[best_offset]
        best_sec    = librosa.frames_to_time(best_offset, sr=sr, hop_length=HOP)

        offsets_sec = {librosa.frames_to_time(k, sr=sr, hop_length=HOP): v
                       for k, v in oc.items()}

        color = TEAL if idx == 0 else '#334455'
        ax.bar(list(offsets_sec.keys()), list(offsets_sec.values()),
               width=0.5, color=color, alpha=0.9)

        if idx == 0:
            ax.set_xlim(best_sec - 12, best_sec + 12)
            ax.axvline(x=best_sec, color='#ff6b6b', linewidth=2,
                       linestyle='--', label=f'{best_count:,} hashes align here')
            leg = ax.legend(fontsize=8)
            leg.get_frame().set_facecolor(DARK_AX)
            for t in leg.get_texts(): t.set_color('#ff6b6b')

        ax.set_ylim(0, best_count * 1.25 if best_count > 0 else 1)
        label = '✓ Correct match' if idx == 0 else '✗ Wrong song'
        ax.set_title(f"{label}: {song}\n(peak={best_count:,}, unique={len(oc)})",
                     color='#ccccdd' if idx == 0 else TEXT_COL, fontsize=9)
        ax.set_xlabel('Time offset (seconds)')
        ax.set_ylabel('# hashes')

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SCORE BAR HTML
# ─────────────────────────────────────────────────────────────────────────────

def score_bars_html(ranked, top_n=5):
    max_score = ranked[0][1][0] if ranked else 1
    rows = ""
    for song, (sc, _) in ranked[:top_n]:
        pct   = int(sc / max_score * 100) if max_score > 0 else 0
        rows += f"""
        <div class="score-row">
          <div class="score-name">{song}</div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%"></div></div>
          <div class="score-val">{sc:,}</div>
        </div>"""
    return f'<div style="margin:1rem 0">{rows}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DB
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(DB_PATH):
    st.error(f"'{DB_PATH}' not found. Place the CSV in the same folder as app.py.")
    st.stop()

with st.spinner("Loading database..."):
    database = load_database(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["◆ LIBRARY", "◎ IDENTIFY", "▦ BATCH"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="subtitle" style="margin-bottom:1rem">LIBRARY</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;padding:1.5rem;
                text-align:center;color:#555577;font-family:Space Mono,monospace;font-size:0.8rem;margin-bottom:2rem">
        Song indexing is managed by the admin.<br>Drop a clip in the Identify tab to test the library.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subtitle" style="margin-bottom:1rem">IN THE DATABASE</div>', unsafe_allow_html=True)

    songs = sorted(database.keys())
    cols  = st.columns(4)
    for i, song in enumerate(songs):
        n_hashes = len(database[song])
        with cols[i % 4]:
            # Mini constellation scatter as a tiny chart
            fig_mini, ax_mini = plt.subplots(figsize=(2.2, 1.4))
            fig_mini.patch.set_facecolor('#0f0f1a')
            ax_mini.set_facecolor('#0f0f1a')
            ax_mini.axis('off')

            # Sample random points to simulate constellation thumbnail
            rng = np.random.default_rng(seed=hash(song) % 9999)
            n   = min(n_hashes, 300)
            xs  = rng.uniform(0, 100, n)
            ys  = rng.uniform(0, 512, n)
            colors_pts = plt.cm.plasma(rng.uniform(0.3, 1.0, n))
            ax_mini.scatter(xs, ys, s=1.5, c=colors_pts, linewidths=0)
            plt.tight_layout(pad=0)

            st.pyplot(fig_mini, use_container_width=True)
            plt.close(fig_mini)
            st.markdown(f"""
            <div class="card-title">{song}</div>
            <div class="card-sub">{n_hashes:,} hashes</div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IDENTIFY
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="subtitle" style="margin-bottom:0.5rem">SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#ffffff;margin-top:0">Identify a clip</h2>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a query clip",
                                 type=["mp3", "wav", "flac", "m4a"],
                                 label_visibility="collapsed")

    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)

        with st.spinner("Fingerprinting..."):
            y, sr, D_db, t_idx, f_idx, clip_hashes, scores, ranked = identify(audio_bytes)

        best_song  = ranked[0][0]
        best_score = ranked[0][1][0]
        runner_up  = ranked[1][1][0] if len(ranked) > 1 else 1
        ratio      = round(best_score / max(runner_up, 1))

        # ── Match result ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="match-box">
          <div class="match-label">MATCH FOUND</div>
          <div class="match-title">{best_song}</div>
          <div class="match-score">
            cluster score <span>{best_score:,}</span> · <span>{ratio:,}×</span> the runner-up
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Candidate scores ──────────────────────────────────────────────────
        st.markdown('<div class="subtitle" style="margin:1.5rem 0 0.75rem 0">CANDIDATE SCORES</div>',
                    unsafe_allow_html=True)
        st.markdown(score_bars_html(ranked), unsafe_allow_html=True)

        st.divider()

        # ── Step 1: Spectrogram + Constellation ───────────────────────────────
        st.markdown("""
        <div class="step-box">
          <div class="step-label">STEP 1 · FEATURE EXTRACTION</div>
          <div class="step-title">From spectrogram to constellation</div>
          <div class="step-desc">
            The clip was converted into a time-frequency map (left); brighter means louder at that
            frequency and moment. From that rich image, only the
            <span>{} most prominent peaks</span> were kept (right).
            Discarding amplitude and phase makes the fingerprint robust to noise.
          </div>
        </div>
        """.format(len(t_idx)), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_spectrogram(D_db, sr), use_container_width=True)
        with col2:
            st.pyplot(plot_constellation(D_db, t_idx, f_idx, sr), use_container_width=True)

        # ── Step 2: Alignment spike ───────────────────────────────────────────
        st.markdown(f"""
        <div class="step-box">
          <div class="step-label">STEP 2 · THE PROOF</div>
          <div class="step-title">The alignment spike</div>
          <div class="step-desc">
            Every matched hash votes for a time offset. Chance matches scatter randomly
            forming a flat noise floor. A genuine match makes them converge:
            <span>{best_score:,} hashes agreed on a single offset</span>.
            That spike cannot be a coincidence.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.pyplot(plot_alignment_spike(scores, ranked, sr), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="subtitle" style="margin-bottom:0.5rem">BATCH</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#ffffff;margin-top:0">Identify many clips at once</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#888899;font-size:0.85rem;margin-bottom:1.5rem;line-height:1.7">
    Upload a set of query clips. Each is identified against the currently indexed library,
    and the results are written to a standardised <code style="color:#00d4a8">results.csv</code>
    with columns <code style="color:#00d4a8">filename</code>, <code style="color:#00d4a8">prediction</code>.
    The prediction is the matched track's filename without its extension.
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader("Upload query clips",
                                    type=["mp3", "wav", "flac", "m4a"],
                                    accept_multiple_files=True,
                                    label_visibility="collapsed")

    if batch_files and st.button("Run batch"):
        results  = []
        progress = st.progress(0, text="Processing...")

        for i, f in enumerate(batch_files):
            try:
                _, _, _, _, _, _, sc, rnk = identify(f.read())
                pred = rnk[0][0]
            except:
                pred = "none"
            results.append({"filename": os.path.splitext(f.name)[0], "prediction": pred})
            progress.progress((i + 1) / len(batch_files),
                               text=f"Processed {i+1}/{len(batch_files)}: {f.name}")

        progress.empty()
        df = pd.DataFrame(results)

        # Results table
        st.markdown('<div class="subtitle" style="margin:1.5rem 0 0.75rem 0">RESULTS</div>',
                    unsafe_allow_html=True)

        rows_html = "".join(
            f'<tr><td>{r["filename"]}</td><td class="pred">{r["prediction"]}</td></tr>'
            for _, r in df.iterrows()
        )
        st.markdown(f"""
        <table class="results-table">
          <thead><tr><th>FILE</th><th>PREDICTION</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        <div style="color:#555577;font-size:0.78rem;margin-top:0.75rem;font-family:Space Mono,monospace">
          {len(df)} / {len(batch_files)} clips matched
        </div>
        """, unsafe_allow_html=True)

        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download results.csv", csv_bytes,
                           "results.csv", "text/csv")