# app/service/activity_service.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from ..db import Database
from ..dao.sit_event_dao import get_sit_events_overlapping
from ..dao.sleep_records_dao import get_sleep_records_overlapping


# ---------- interval helpers ----------
def _clip_interval(start: datetime, end: datetime, win_start: datetime, win_end: datetime) -> Tuple[datetime, datetime] | None:
    s = max(start, win_start)
    e = min(end, win_end)
    if e <= s:
        return None
    return (s, e)

def _merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """把重疊或相接的區段合併，避免重複計算"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged

def _sum_secs(intervals: List[Tuple[datetime, datetime]]) -> int:
    return sum(int((e - s).total_seconds()) for s, e in intervals)

def _intervals_intersection(a: List[Tuple[datetime, datetime]], b: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """兩組（已排序/合併過）的區段交集"""
    i = j = 0
    out: List[Tuple[datetime, datetime]] = []
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            out.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return out

def _bout_stats(intervals: List[Tuple[datetime, datetime]]) -> Dict[str, Any]:
    """連續區段統計：最長一段（分鐘）與段數"""
    if not intervals:
        return {"count": 0, "longest_minutes": 0.0}
    longest = max((e - s).total_seconds() for s, e in intervals)
    return {"count": len(intervals), "longest_minutes": round(longest / 60.0, 1)}


# ---------- service entry ----------
async def get_daily_activity(user_id: int, date_str: str) -> Dict[str, Any]:
    """
    以 user_id + date（YYYY-MM-DD）計算當天 00:00–24:00 的活動量：
    - 睡眠 / 坐著 秒數（自動處理跨日 & NULL 結束時間）
    - 醒著時間、活動分鐘、久坐比例、活動分數
    - 區段明細與連續久坐統計
    """
    day_start = datetime.fromisoformat(date_str).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    async with Database.connection() as conn:
        sit_rows = await get_sit_events_overlapping(conn, user_id, day_start, day_end)
        slp_rows = await get_sleep_records_overlapping(conn, user_id, day_start, day_end)

    # 裁切成落在當日窗內的實際區段
    sit_intervals: List[Tuple[datetime, datetime]] = []
    for r in sit_rows:
        s = r["start_at"]
        e = r["end_at"] or day_end
        c = _clip_interval(s, e, day_start, day_end)
        if c:
            sit_intervals.append(c)

    sleep_intervals: List[Tuple[datetime, datetime]] = []
    for r in slp_rows:
        s = r["sleep_time"]
        e = r["wake_time"] or day_end
        c = _clip_interval(s, e, day_start, day_end)
        if c:
            sleep_intervals.append(c)

    # 合併避免重疊重算
    sit_merged = _merge_intervals(sit_intervals)
    sleep_merged = _merge_intervals(sleep_intervals)

    # 各自秒數與交集
    sit_secs = _sum_secs(sit_merged)
    sleep_secs = _sum_secs(sleep_merged)
    overlap_sit_sleep_secs = _sum_secs(_intervals_intersection(sit_merged, sleep_merged))

    # 指標
    day_secs = 24 * 60 * 60
    awake_secs = max(0, day_secs - sleep_secs)
    active_secs = max(0, day_secs - sleep_secs - sit_secs + overlap_sit_sleep_secs)
    sedentary_ratio = (sit_secs / awake_secs) if awake_secs > 0 else 0.0
    activity_score = round(100 * (1 - sedentary_ratio))

    # 連續久坐統計（可做提醒/排行榜）
    sit_bout = _bout_stats(sit_merged)

        # 判斷是否有睡眠資料（當日窗內）
    sleep_data_missing = (len(sleep_merged) == 0) or (sleep_secs == 0)

    # 需要對前端提示的訊息
    messages = []
    if sleep_data_missing:
        messages.append("當天沒有睡眠資料")

    return {
        "user_id": user_id,
        "date": date_str,
        "window": {"start": day_start.isoformat(), "end": day_end.isoformat()},
        "totals": {
            "sleep_minutes": round(sleep_secs / 60.0, 1),
            "sit_minutes": round(sit_secs / 60.0, 1),
            "overlap_sit_sleep_minutes": round(overlap_sit_sleep_secs / 60.0, 1),
            "awake_minutes": round(awake_secs / 60.0, 1),
            "active_minutes": round(active_secs / 60.0, 1),
            "sedentary_ratio": round(sedentary_ratio, 3),
            "activity_score": activity_score
        },
        "details": {
            "sleep_intervals": [(s.isoformat(), e.isoformat()) for s, e in sleep_merged],
            "sit_intervals": [(s.isoformat(), e.isoformat()) for s, e in sit_merged],
            "sit_bout": sit_bout
        },
        "flags": {
            "sleep_data_missing": sleep_data_missing
        },
        "messages": messages
    }

def _week_range(date_str: str, week_start: str) -> Tuple[datetime, datetime, str]:
    """
    回傳該週 [start, end)；week_start='sun' 表週日為第一天，'mon' 表週一為第一天。
    Python weekday(): Mon=0..Sun=6
    """
    d0 = datetime.fromisoformat(date_str).replace(hour=0, minute=0, second=0, microsecond=0)
    w = d0.weekday()  # Mon=0..Sun=6
    if week_start.lower() == "sun":
        # 當週週日：d0 往回 (w+1)%7 天
        delta_days = (w + 1) % 7
        week_first = d0 - timedelta(days=delta_days)
        start_wd = "sun"
    else:
        # 當週週一：d0 往回 w 天
        week_first = d0 - timedelta(days=w)
        start_wd = "mon"
    return week_first, week_first + timedelta(days=7), start_wd

def _day_activity_from_rows(
    day_start: datetime,
    day_end: datetime,
    sit_rows: List[Dict[str, Any]],
    slp_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # --- 裁切 ---
    sit_intervals = []
    for r in sit_rows:
        s = r["start_at"]; e = r["end_at"] or day_end
        c = _clip_interval(s, e, day_start, day_end)
        if c: sit_intervals.append(c)

    sleep_intervals = []
    for r in slp_rows:
        s = r["sleep_time"]; e = r["wake_time"] or day_end
        c = _clip_interval(s, e, day_start, day_end)
        if c: sleep_intervals.append(c)

    # --- 合併 / 計算 ---
    sit_m   = _merge_intervals(sit_intervals)
    sleep_m = _merge_intervals(sleep_intervals)

    sit_secs   = _sum_secs(sit_m)
    sleep_secs = _sum_secs(sleep_m)
    inter_secs = _sum_secs(_intervals_intersection(sit_m, sleep_m))

    day_secs = 24 * 60 * 60
    awake_secs = max(0, day_secs - sleep_secs)
    active_secs = max(0, day_secs - sleep_secs - sit_secs + inter_secs)
    sedentary_ratio = (sit_secs / awake_secs) if awake_secs > 0 else 0.0
    activity_score  = round(100 * (1 - sedentary_ratio))

    sit_bout = _bout_stats(sit_m)

    sleep_data_missing = (len(sleep_m) == 0) or (sleep_secs == 0)
    messages = []
    if sleep_data_missing:
        messages.append("當天沒有睡眠資料")

    return {
        "window": {"start": day_start.isoformat(), "end": day_end.isoformat()},
        "totals": {
            "sleep_minutes": round(sleep_secs / 60.0, 1),
            "sit_minutes": round(sit_secs / 60.0, 1),
            "overlap_sit_sleep_minutes": round(inter_secs / 60.0, 1),
            "awake_minutes": round(awake_secs / 60.0, 1),
            "active_minutes": round(active_secs / 60.0, 1),
            "sedentary_ratio": round(sedentary_ratio, 3),
            "activity_score": activity_score
        },
        "details": {
            "sleep_intervals": [(s.isoformat(), e.isoformat()) for s, e in sleep_m],
            "sit_intervals":   [(s.isoformat(), e.isoformat()) for s, e in sit_m],
            "sit_bout": sit_bout
        },
        "flags": {
            "sleep_data_missing": sleep_data_missing,
            "complete": (not sleep_data_missing)
        },
        "messages": messages
    }

async def get_weekly_activity(user_id: int, date_str: str, week_start: str = "mon") -> Dict[str, Any]:
    week_s, week_e, start_wd = _week_range(date_str, week_start)

    async with Database.connection() as conn:
        # 一次抓整週的 rows
        sit_rows_week = await get_sit_events_overlapping(conn, user_id, week_s, week_e)
        slp_rows_week = await get_sleep_records_overlapping(conn, user_id, week_s, week_e)

    # ★ 週級別偵錯列印（協助排查）
    print(f"[WEEK] user={user_id}, week={week_s}..{week_e}, sit_rows={len(sit_rows_week)}, sleep_rows={len(slp_rows_week)}")

    daily: List[Dict[str, Any]] = []
    week_tot_sleep = week_tot_sit = week_tot_active = 0.0
    missing_days: List[str] = []

    for i in range(7):
        ds = week_s + timedelta(days=i)
        de = ds + timedelta(days=1)
        day_stats = _day_activity_from_rows(ds, de, sit_rows_week, slp_rows_week)
        day_stats["date"] = ds.date().isoformat()

        week_tot_sleep  += day_stats["totals"]["sleep_minutes"]
        week_tot_sit    += day_stats["totals"]["sit_minutes"]
        week_tot_active += day_stats["totals"]["active_minutes"]

        if not day_stats["flags"]["complete"]:
            missing_days.append(day_stats["date"])

        daily.append(day_stats)

    complete_week = (len(missing_days) == 0)
    week_awake_minutes = 7 * 24 * 60 - week_tot_sleep
    week_sedentary_ratio = (week_tot_sit / week_awake_minutes) if week_awake_minutes > 0 else 0.0
    week_activity_score = round(100 * (1 - week_sedentary_ratio))

    empty_week = (len(sit_rows_week) == 0 and len(slp_rows_week) == 0)

    out = {
        "user_id": user_id,
        "anchor_date": date_str,
        "week": {
            "start": week_s.isoformat(),
            "end": week_e.isoformat(),
            "start_weekday": start_wd
        },
        "daily": daily,
        "totals": {
            "sleep_minutes": round(week_tot_sleep, 1),
            "sit_minutes": round(week_tot_sit, 1),
            "active_minutes": round(week_tot_active, 1),
            "awake_minutes": round(week_awake_minutes, 1),
            "sedentary_ratio": round(week_sedentary_ratio, 3),
            "activity_score": week_activity_score
        },
        "flags": {
            "complete_week": complete_week,
            "missing_days": missing_days,
            "empty_week": empty_week
        },
        "messages": ([] if not empty_week else ["這一週完全沒有任何坐著/睡眠資料"])
    }
    return out