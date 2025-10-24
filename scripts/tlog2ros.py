#!/usr/bin/env python3
"""
tlog_dive_extractor.py

- Read every *.tlog in a user-chosen folder
- Keep ATTITUDE rows while depth >= 15 m
- Build dives ≥ 30 min (max 10 s gap)
- Write CSV + per-dive depth plots in the parent directory
- Produce a CSV summary of all segments (kept/dropped) with metrics
  (no location info) and print it on-screen

All timestamps are now output in UTC.
"""

from __future__ import annotations
import csv, sys, re, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from pymavlink import mavutil  # type: ignore

# ───────── constants ─────────
DEPTH_MIN   = 10.0       # m
MIN_DIVE_S  = 600        # s
UTC         = timezone.utc

def sanitize(tag: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", tag)
    if not clean:
        raise ValueError("Dive location must contain letters/numbers.")
    return clean.upper()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def boot_offset(tlog: Path) -> float | None:
    m = mavutil.mavlink_connection(str(tlog))
    while True:
        msg = m.recv_match(type="SYSTEM_TIME", blocking=False)
        if msg is None:
            break
        if msg.time_unix_usec:
            m.close()
            return msg.time_unix_usec/1e6 - msg.time_boot_ms/1e3
    m.close()
    return None

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs) if xs else float("nan")

def main() -> None:
    folder = Path(input("Folder with .tlog files: ").strip()).expanduser()
    if not folder.is_dir():
        sys.exit("Folder not found.")
    tag = sanitize(input("Dive location (one word): "))

    tlogs = sorted(folder.glob("*.tlog"))
    if not tlogs:
        sys.exit("No .tlog files.")

    all_rows: List[Dict] = []
    global_ctr = Counter()

    # parse all .tlogs
    for tlog in tlogs:
        print(f"Reading {tlog.name} …")
        off = boot_offset(tlog)
        m = mavutil.mavlink_connection(str(tlog))
        m.wait_heartbeat()

        p0_air = depth = temp = None
        while (msg := m.recv_match(blocking=False)):
            if not hasattr(msg, "time_boot_ms"):
                continue
            mt = msg.get_type()

            if mt == "SCALED_PRESSURE":
                p0_air = msg.press_abs * 100
            elif mt == "SCALED_PRESSURE2" and p0_air:
                depth = (msg.press_abs * 100 - p0_air) / (1000 * 9.80665)
                depth = max(0.0, depth)
                temp = msg.temperature / 100

            elif mt == "ATTITUDE":
                global_ctr["att_total"] += 1
                if depth is None:
                    global_ctr["no_depth"] += 1
                    continue
                if depth < DEPTH_MIN:
                    global_ctr["shallow"] += 1
                    continue

                tb = msg.time_boot_ms
                if off is not None:
                    epoch = off + tb/1000
                elif getattr(msg, "_timestamp", 0):
                    epoch = msg._timestamp
                else:
                    epoch = tb/1000.0

                tsu = int(epoch * 1e6)
                all_rows.append({
                    "unix_time_us":   tsu,
                    "timestamp_utc":  iso_utc(epoch),
                    "roll_deg":       round(math.degrees(msg.roll), 3),
                    "pitch_deg":      round(math.degrees(msg.pitch), 3),
                    "heading_deg":    round((math.degrees(msg.yaw)+360) % 360, 3),
                    "depth_m":        round(depth, 2),
                    "water_temp_c":   round(temp, 2) if temp is not None else None,
                    "dive_num":       0
                })
                global_ctr["deep_rows"] += 1

        m.close()

    if not all_rows:
        sys.exit("No deep attitude rows found.")

    # ───── Raw stats ─────
    print("\n=== Raw processing stats ===")
    print(f"  ATTITUDE msgs total: {global_ctr['att_total']}")
    print(f"  Dropped no-depth   : {global_ctr['no_depth']}")
    print(f"  Dropped shallow    : {global_ctr['shallow']}")
    print(f"  Kept deep rows     : {global_ctr['deep_rows']}")

    # segment by gaps >10s
    all_rows.sort(key=lambda r: r["unix_time_us"])
    segments = []
    start = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["unix_time_us"] - all_rows[i-1]["unix_time_us"] > 10_000_000:
            segments.append((start, i-1))
            start = i
    segments.append((start, len(all_rows)-1))

    # build summary and kept-rows
    summary = []
    kept_rows: List[Dict] = []
    dive_id = 0

    for seg_idx, (s, e) in enumerate(segments, start=1):
        t0 = all_rows[s]["unix_time_us"]
        t1 = all_rows[e]["unix_time_us"]
        dur_s = (t1 - t0) / 1e6
        orig_count = e - s + 1

        if dur_s >= MIN_DIVE_S:
            dive_id += 1
            status = "kept"
            seg = all_rows[s:e+1]
            for r in seg:
                r["dive_num"] = dive_id
            kept_rows.extend(seg)
            depths = [r["depth_m"] for r in seg]
            rows_kept = len(seg)
            mean_dep = mean(depths)
            max_dep = max(depths)
            dive_label = f"LONMS_{dive_id:03d}"
        else:
            status = "dropped"
            rows_kept = 0
            mean_dep = ""
            max_dep = ""
            dive_label = ""

        summary.append({
            "segment_idx":   seg_idx,
            "start_utc":     iso_utc(t0/1e6),
            "end_utc":       iso_utc(t1/1e6),
            "duration_min":  f"{dur_s/60:.1f}",
            "status":        status,
            "reason":        f"{dur_s/60:.1f}min<{MIN_DIVE_S/60:.1f}min" if status=="dropped" else "",
            "dive_id":       dive_label,
            "rows_original": orig_count,
            "rows_kept":     rows_kept,
            "mean_depth_m":  mean_dep,
            "max_depth_m":   max_dep
        })

    total_segs = len(summary)
    kept_segs  = sum(1 for x in summary if x["status"]=="kept")
    drop_segs  = total_segs - kept_segs
    print(f"\nSegments: {total_segs}, kept: {kept_segs}, dropped: {drop_segs}")

    # write segment summary CSV
    sum_csv = folder.parent / f"{tag.lower()}_dive_summary.csv"
    fields = [
        "segment_idx","start_utc","end_utc","duration_min",
        "status","reason","dive_id",
        "rows_original","rows_kept",
        "mean_depth_m","max_depth_m"
    ]
    with sum_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"\nSegment summary written to: {sum_csv}")

    # write kept-rows CSV
    if not kept_rows:
        sys.exit("No dives kept.")
    kept_csv = folder.parent / f"{tag.lower()}_rovlog.csv"
    with kept_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_rows)
    print(f"Kept rows CSV written to: {kept_csv}")

    # per-dive depth plots
    by_dive = defaultdict(list)
    for r in kept_rows:
        by_dive[r["dive_num"]].append(r)

    for did, sub in by_dive.items():
        times = [datetime.fromtimestamp(r["unix_time_us"]/1e6, tz=UTC) for r in sub]
        deps  = [r["depth_m"] for r in sub]
        t0 = times[0]
        dur_min   = (times[-1] - t0).total_seconds() / 60
        date_str  = t0.strftime("%Y%m%d")      # for filename
        date_label= t0.strftime("%Y-%m-%d")    # for title
        label     = f"LONMS_{did:03d}"

        plt.figure()
        plt.plot(times, deps)
        plt.title(f"{label} – {date_label} – {dur_min:.1f} min")
        plt.xlabel("UTC")
        plt.ylabel("Depth (m)")
        plt.gca().invert_yaxis()
        plt.gcf().autofmt_xdate()

        outfn = folder.parent / f"{label}_{date_str}_depth.png"
        plt.savefig(outfn, dpi=150)
        plt.close()

    print("Depth plots saved to parent folder as LONMS_###_YYYYMMDD_depth.png")

if __name__ == "__main__":
    main()