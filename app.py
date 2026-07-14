"""Fantasy football draft optimizer — Streamlit + pandas + SQLite.

Run with: streamlit run app.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "draft.db"
SAMPLE_CSV = APP_DIR / "sample_projections.csv"

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
DEFAULT_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}

NAME_ALIASES = {"player", "player name", "name", "full name"}
POS_ALIASES = {"pos", "position"}
TEAM_ALIASES = {"team", "tm", "nfl team"}
POINTS_ALIASES = {
    "proj", "projection", "projected points", "proj points", "proj pts",
    "fpts", "points", "pts", "fantasy points",
}
REC_ALIASES = {"rec", "receptions", "projected receptions", "rec proj"}

POSITION_FIXES = {"D/ST": "DST", "DEF": "DST", "D": "DST", "PK": "K"}

SCORING_REC_BONUS = {"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}

TIER_COLORS = {
    1: "#00e676", 2: "#40c4ff", 3: "#b388ff",
    4: "#ffd740", 5: "#ff9e40", 6: "#9e9e9e",
}


# ---------------------------------------------------------------- data loading

def norm_header(header):
    """Lowercase a header and collapse punctuation/whitespace so messy
    variants like ' Proj._Points ' match 'proj points'."""
    header = re.sub(r"[^a-z0-9 ]", " ", str(header).lower())
    return re.sub(r"\s+", " ", header).strip()


def find_column(df, aliases):
    for col in df.columns:
        if norm_header(col) in aliases:
            return col
    for col in df.columns:
        if any(alias in norm_header(col) for alias in aliases):
            return col
    return None


def normalize_position(value):
    pos = re.sub(r"[0-9]", "", str(value).upper().strip())
    return POSITION_FIXES.get(pos, pos)


def load_projections(source):
    """Read a projections CSV into a tidy DataFrame, tolerating messy headers.

    Returns columns: player, position, team, base_points, receptions, key.
    """
    raw = pd.read_csv(source)

    name_col = find_column(raw, NAME_ALIASES)
    pos_col = find_column(raw, POS_ALIASES)
    pts_col = find_column(raw, POINTS_ALIASES)
    team_col = find_column(raw, TEAM_ALIASES)
    rec_col = find_column(raw, REC_ALIASES)

    missing = [label for label, col in
               [("player name", name_col), ("position", pos_col), ("projected points", pts_col)]
               if col is None]
    if missing:
        raise ValueError(f"Could not find column(s) for: {', '.join(missing)}. "
                         f"Headers found: {list(raw.columns)}")

    df = pd.DataFrame({
        "player": raw[name_col].astype(str).str.strip(),
        "position": raw[pos_col].map(normalize_position),
        "team": raw[team_col].astype(str).str.strip().str.upper() if team_col is not None else "",
        "base_points": pd.to_numeric(raw[pts_col], errors="coerce"),
        "receptions": pd.to_numeric(raw[rec_col], errors="coerce").fillna(0.0)
                      if rec_col is not None else 0.0,
    })
    df.attrs["has_receptions"] = rec_col is not None

    df = df[df["position"].isin(POSITIONS)]
    df = df.dropna(subset=["base_points"])
    df = df[df["player"] != ""]
    df["key"] = (df["player"] + "|" + df["position"] + "|" + df["team"]).str.lower()
    df = df.drop_duplicates(subset="key").reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable player rows found in the CSV.")
    return df


# ---------------------------------------------------------------- core math

def apply_scoring(df, scoring):
    """Points under the chosen scoring. base_points is treated as standard
    scoring; PPR/Half add a per-reception bonus when receptions are known."""
    bonus = SCORING_REC_BONUS[scoring]
    df["points"] = (df["base_points"] + bonus * df["receptions"]).round(1)
    return df


def compute_vor(df, num_teams, starters):
    """VOR = points minus the last drafted starter at that position
    (player ranked num_teams x starters within the position)."""
    baselines = {}
    for pos, grp in df.groupby("position"):
        pts = grp["points"].sort_values(ascending=False)
        n_starters = num_teams * starters.get(pos, 0)
        idx = min(max(n_starters, 1), len(pts)) - 1
        baselines[pos] = pts.iloc[idx]
    df["vor"] = (df["points"] - df["position"].map(baselines)).round(1)
    return df, baselines


def assign_tiers(df, gap_mult=1.5, max_tiers=6):
    """Tier players within each position: a new tier starts when the drop to
    the next player exceeds gap_mult x the median gap for that position."""
    df["tier"] = 1
    for pos, grp in df.groupby("position"):
        ordered = grp.sort_values("points", ascending=False)
        pts = ordered["points"].to_numpy()
        if len(pts) < 2:
            continue
        gaps = pts[:-1] - pts[1:]
        threshold = max(gap_mult * float(pd.Series(gaps).median()), 0.1)
        tier = 1
        tiers = [1]
        for gap in gaps:
            if gap > threshold and tier < max_tiers:
                tier += 1
            tiers.append(tier)
        df.loc[ordered.index, "tier"] = tiers
    return df


# ---------------------------------------------------------------- persistence

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            player TEXT NOT NULL,
            position TEXT NOT NULL,
            mine INTEGER NOT NULL DEFAULT 0,
            picked_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def get_picks():
    with db() as conn:
        return pd.read_sql_query("SELECT * FROM picks ORDER BY id", conn)


def draft_player(key, player, position, mine):
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO picks (key, player, position, mine) VALUES (?, ?, ?, ?)",
            (key, player, position, int(mine)),
        )


def undo_last_pick():
    with db() as conn:
        conn.execute("DELETE FROM picks WHERE id = (SELECT MAX(id) FROM picks)")


def reset_draft():
    with db() as conn:
        conn.execute("DELETE FROM picks")


# ---------------------------------------------------------------- UI helpers

CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp { background: #0a0a0a; }
h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }

.stat-strip { display: flex; gap: 12px; margin-bottom: 8px; }
.stat-card {
    flex: 1; background: #111111; border: 1px solid #262626;
    border-radius: 12px; padding: 14px 18px;
}
.stat-label { color: #888; font-size: 0.75rem; text-transform: uppercase;
              letter-spacing: 0.08em; font-weight: 700; }
.stat-value { color: #D4AF37; font-size: 1.7rem; font-weight: 800; line-height: 1.2; }
.stat-value.green { color: #00e676; font-size: 1.25rem; }

.best-card {
    background: linear-gradient(135deg, #0d1f14 0%, #111111 60%);
    border: 1px solid #00e676; border-radius: 16px;
    padding: 22px 26px; margin-bottom: 14px;
}
.best-eyebrow { color: #00e676; font-size: 0.8rem; font-weight: 800;
                text-transform: uppercase; letter-spacing: 0.12em; }
.best-name { color: #ffffff; font-size: 2.1rem; font-weight: 800; line-height: 1.15; }
.best-meta { color: #aaa; font-size: 1rem; margin-top: 2px; }
.best-numbers { display: flex; gap: 28px; margin-top: 12px; }
.best-num-label { color: #888; font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 0.08em; font-weight: 700; }
.best-num-value { color: #D4AF37; font-size: 1.5rem; font-weight: 800; }
.best-num-value.green { color: #00e676; }

.tier-badge {
    display: inline-block; padding: 1px 9px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 800; color: #0a0a0a;
}
.player-chip {
    display: inline-block; background: #111111; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 4px 10px; margin: 3px 4px 3px 0;
    font-size: 0.85rem; color: #eee;
}
.player-chip .pts { color: #D4AF37; font-weight: 700; }
.player-chip.taken { opacity: 0.35; text-decoration: line-through; }

.board-row { border-bottom: 1px solid #1c1c1c; }
.vor-val { color: #00e676; font-weight: 800; }
.roster-slot { background: #111111; border: 1px solid #262626; border-radius: 8px;
               padding: 6px 12px; margin: 4px 0; font-size: 0.9rem; }
.roster-slot.empty { color: #555; border-style: dashed; }
.roster-slot.bench { border-color: #3a3a2a; }
</style>
"""


def tier_badge(tier):
    color = TIER_COLORS.get(int(tier), "#9e9e9e")
    return f'<span class="tier-badge" style="background:{color}">T{int(tier)}</span>'


def stat_strip(n_drafted, n_mine, best):
    best_txt = f"{best['player']} ({best['position']})" if best is not None else "—"
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-card"><div class="stat-label">Players drafted</div>
        <div class="stat-value">{n_drafted}</div></div>
      <div class="stat-card"><div class="stat-label">My picks</div>
        <div class="stat-value">{n_mine}</div></div>
      <div class="stat-card"><div class="stat-label">Best available</div>
        <div class="stat-value green">{best_txt}</div></div>
    </div>
    """, unsafe_allow_html=True)


def best_pick_card(player, deweighted):
    note = ('<div class="best-meta">Position starters filled — best value elsewhere '
            'was still highest.</div>') if deweighted else ""
    st.markdown(f"""
    <div class="best-card">
      <div class="best-eyebrow">Best pick now</div>
      <div class="best-name">{player['player']}</div>
      <div class="best-meta">{player['position']} · {player['team']} · Tier {int(player['tier'])}</div>
      <div class="best-numbers">
        <div><div class="best-num-label">Projected</div>
          <div class="best-num-value">{player['points']:.1f}</div></div>
        <div><div class="best-num-label">VOR</div>
          <div class="best-num-value green">{player['vor']:+.1f}</div></div>
      </div>{note}
    </div>
    """, unsafe_allow_html=True)


def draft_buttons(row, key_prefix):
    """Two small buttons: draft to my team / mark taken by another team."""
    col_mine, col_taken = st.columns(2)
    if col_mine.button("✓ Mine", key=f"{key_prefix}_mine_{row['key']}",
                       help="Draft to my roster"):
        draft_player(row["key"], row["player"], row["position"], mine=True)
        st.rerun()
    if col_taken.button("✗ Taken", key=f"{key_prefix}_taken_{row['key']}",
                        help="Drafted by another team"):
        draft_player(row["key"], row["player"], row["position"], mine=False)
        st.rerun()


# ---------------------------------------------------------------- sections

def render_draft_board(available):
    search = st.text_input("Search players", placeholder="Type a name…")
    filt_col, limit_col = st.columns([3, 1])
    pos_filter = filt_col.multiselect("Positions", POSITIONS, default=POSITIONS)
    limit = limit_col.slider("Rows", 10, len(available) if len(available) > 10 else 10,
                             min(40, len(available)))

    view = available[available["position"].isin(pos_filter)]
    if search:
        view = view[view["player"].str.contains(search, case=False, regex=False)]
    view = view.head(limit)

    header = st.columns([0.6, 2.4, 0.6, 0.7, 0.9, 0.9, 0.7, 1.6])
    for col, label in zip(header, ["Rank", "Player", "Pos", "Team", "Proj", "VOR", "Tier", "Draft"]):
        col.markdown(f"**{label}**")

    for _, row in view.iterrows():
        cols = st.columns([0.6, 2.4, 0.6, 0.7, 0.9, 0.9, 0.7, 1.6])
        cols[0].markdown(f"{int(row['rank'])}")
        cols[1].markdown(f"**{row['player']}**")
        cols[2].markdown(row["position"])
        cols[3].markdown(row["team"] or "—")
        cols[4].markdown(f"{row['points']:.1f}")
        cols[5].markdown(f'<span class="vor-val">{row["vor"]:+.1f}</span>',
                         unsafe_allow_html=True)
        cols[6].markdown(tier_badge(row["tier"]), unsafe_allow_html=True)
        with cols[7]:
            draft_buttons(row, "board")

    with st.expander("Full sortable table (all available players)"):
        st.dataframe(
            available[["rank", "player", "position", "team", "points", "vor", "tier"]]
            .rename(columns={"points": "proj"}),
            use_container_width=True, hide_index=True,
        )


def render_tiers(board):
    pos_choices = st.multiselect("Show positions", POSITIONS, default=POSITIONS,
                                 key="tier_pos")
    for pos in pos_choices:
        grp = board[board["position"] == pos].sort_values("points", ascending=False)
        if grp.empty:
            continue
        st.subheader(pos)
        for tier, tgrp in grp.groupby("tier"):
            chips = "".join(
                f'<span class="player-chip{" taken" if r["drafted"] else ""}">'
                f'{r["player"]} <span class="pts">{r["points"]:.1f}</span></span>'
                for _, r in tgrp.iterrows()
            )
            st.markdown(f'{tier_badge(tier)}&nbsp; {chips}', unsafe_allow_html=True)


def render_draft_mode(available, board, picks, starters):
    left, right = st.columns([2, 1])

    with left:
        if available.empty:
            st.info("Every player has been drafted.")
        else:
            best = available.sort_values("adj_vor", ascending=False).iloc[0]
            best_pick_card(best, deweighted=bool(best["deweighted"]))
            bcol1, bcol2 = st.columns(2)
            if bcol1.button("✓ Draft to my team", type="primary", width="stretch"):
                draft_player(best["key"], best["player"], best["position"], mine=True)
                st.rerun()
            if bcol2.button("✗ Mark taken", width="stretch"):
                draft_player(best["key"], best["player"], best["position"], mine=False)
                st.rerun()

        st.subheader("Best available by position")
        for pos in POSITIONS:
            pool = available[available["position"] == pos]
            if pool.empty:
                continue
            row = pool.sort_values("vor", ascending=False).iloc[0]
            cols = st.columns([0.7, 2.4, 0.9, 0.9, 0.7, 1.6])
            cols[0].markdown(f"**{pos}**")
            cols[1].markdown(row["player"])
            cols[2].markdown(f"{row['points']:.1f}")
            cols[3].markdown(f'<span class="vor-val">{row["vor"]:+.1f}</span>',
                             unsafe_allow_html=True)
            cols[4].markdown(tier_badge(row["tier"]), unsafe_allow_html=True)
            with cols[5]:
                draft_buttons(row, "bypos")

    with right:
        st.subheader("My roster")
        mine = picks[picks["mine"] == 1]
        for pos in POSITIONS:
            n_slots = starters.get(pos, 0)
            names = mine[mine["position"] == pos]["player"].tolist()
            for i in range(max(n_slots, len(names))):
                if i < len(names):
                    bench = ' bench' if i >= n_slots else ''
                    label = " (bench)" if i >= n_slots else ""
                    st.markdown(f'<div class="roster-slot{bench}">'
                                f'<b>{pos}</b> · {names[i]}{label}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="roster-slot empty">{pos} · empty</div>',
                                unsafe_allow_html=True)

        st.subheader("Draft log")
        if st.button("↩ Undo last pick", width="stretch", disabled=picks.empty):
            undo_last_pick()
            st.rerun()
        if not picks.empty:
            log = picks.sort_values("id", ascending=False).head(10)
            for _, p in log.iterrows():
                who = "🟢 me" if p["mine"] else "⚪ other"
                st.markdown(f"{int(p['id'])}. **{p['player']}** ({p['position']}) — {who}")


# ---------------------------------------------------------------- main

def main():
    st.set_page_config(page_title="Draft Optimizer", page_icon="🏈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- sidebar: data source + league settings
    with st.sidebar:
        st.title("🏈 Draft Optimizer")
        st.header("Data")
        uploaded = st.file_uploader("Upload projections CSV", type="csv")
        if st.button("Use sample data", width="stretch"):
            st.session_state["use_sample"] = True

        st.header("League settings")
        num_teams = st.number_input("Teams", min_value=4, max_value=20, value=12)
        scoring = st.radio("Scoring", ["PPR", "Half PPR", "Standard"], horizontal=True)
        with st.expander("Roster starters"):
            starters = {pos: st.number_input(pos, min_value=0, max_value=4,
                                             value=DEFAULT_STARTERS[pos],
                                             key=f"starters_{pos}")
                        for pos in POSITIONS}

        st.header("Draft")
        if st.button("Reset draft", width="stretch"):
            reset_draft()
            st.rerun()

    # ---- data source
    source = None
    if uploaded is not None:
        source = uploaded
    elif st.session_state.get("use_sample"):
        source = SAMPLE_CSV

    if source is None:
        st.title("Fantasy Football Draft Optimizer")
        st.write("Upload a projections CSV in the sidebar, or start instantly:")
        if st.button("Use sample data", type="primary"):
            st.session_state["use_sample"] = True
            st.rerun()
        st.stop()

    try:
        df = load_projections(source)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    if not df.attrs["has_receptions"] and scoring != "Standard":
        st.sidebar.caption("No receptions column in this CSV — scoring toggle has no effect.")

    # ---- pipeline: scoring -> VOR -> tiers -> merge draft state
    df = apply_scoring(df, scoring)
    df, baselines = compute_vor(df, num_teams, starters)
    df = assign_tiers(df)

    picks = get_picks()
    df["drafted"] = df["key"].isin(picks["key"])
    board = df.sort_values("vor", ascending=False).reset_index(drop=True)
    board["rank"] = board.index + 1

    available = board[~board["drafted"]].copy()

    # de-weight positions whose starters I've already filled
    my_counts = picks[picks["mine"] == 1]["position"].value_counts()
    filled = {pos for pos in POSITIONS
              if starters.get(pos, 0) > 0 and my_counts.get(pos, 0) >= starters[pos]}
    available["deweighted"] = available["position"].isin(filled)
    available["adj_vor"] = available["vor"].where(~available["deweighted"],
                                                  available["vor"] * 0.4)

    # ---- header stat strip
    best_overall = (available.sort_values("adj_vor", ascending=False).iloc[0]
                    if not available.empty else None)
    stat_strip(len(picks), int((picks["mine"] == 1).sum()), best_overall)

    tab_board, tab_tiers, tab_draft = st.tabs(["📋 Draft Board", "🎨 Tiers", "🎯 Draft Mode"])
    with tab_board:
        render_draft_board(available)
    with tab_tiers:
        render_tiers(board)
    with tab_draft:
        render_draft_mode(available, board, picks, starters)


if __name__ == "__main__":
    main()
