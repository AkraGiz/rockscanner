from core.smooth import get_zones


def mark_wood_adjacent_as_mechanical(zones):
    """
    Any fractured/rubble zone immediately adjacent to a wood block is
    forced to mechanical origin.

    Rationale: wood blocks mark drill-run boundaries.  The rock at the
    start/end of a run breaks mechanically due to drill dynamics — it is
    never a natural discontinuity.

    Sets p_nat=0.0, p_mec=1.0, origin_conf='alta' and adds
    origin_det['forced'] = 'adjacent_to_wood' for traceability.
    Zones that have not yet been enriched (no 'p_nat' key) are also
    updated so the bypass-RQD logic works correctly.
    """
    result = [dict(z) for z in zones]
    wood_idx = [i for i, z in enumerate(result) if z["label"] == "wood"]

    for wi in wood_idx:
        for ni in (wi - 1, wi + 1):
            if 0 <= ni < len(result) and result[ni]["label"] in ("fractured", "rubble"):
                det = dict(result[ni].get("origin_det") or {})
                det["forced"] = "adjacent_to_wood"
                result[ni].update({
                    "p_nat":       0.0,
                    "p_mec":       1.0,
                    "origin_conf": "alta",
                    "origin_det":  det,
                })
    return result


def merge_labels_into_zones(labels, windows, img_width, row_length_cm=None):
    """
    Collapse consecutive same-label windows into named zones with positions.

    Returns list of dicts:
      label, x_start, x_end, pct_start, pct_end, cm_start*, cm_end*
    (*None when row_length_cm is not provided)
    """
    if not labels or not windows:
        return []

    raw_zones = get_zones(labels)   # [(label_str, win_start, win_end), ...]

    zones = []
    for label, win_start, win_end in raw_zones:
        x_start = windows[win_start][0]
        x_end   = windows[min(win_end - 1, len(windows) - 1)][1]

        pct_start = x_start / img_width
        pct_end   = x_end   / img_width

        zone = {
            "label":     label,
            "x_start":   x_start,
            "x_end":     x_end,
            "pct_start": pct_start,
            "pct_end":   pct_end,
            "cm_start":  pct_start * row_length_cm if row_length_cm else None,
            "cm_end":    pct_end   * row_length_cm if row_length_cm else None,
        }
        zones.append(zone)

    return zones


def filter_short_intact(zones, min_intact_cm, bypass_mechanical=False):
    """
    Reclassify intact zones shorter than min_intact_cm as fractured.

    bypass_mechanical=True  (RQD mode)
    -----------------------------------
    Mechanical fracture zones (p_mec > p_nat, stored in the zone dict after
    enrichment) are treated as transparent connectors — they don't break an
    intact run.

    A "run" is a maximal sequence of consecutive zones that are either intact
    or mechanical fractures.  The run's total span (start of first zone →
    end of last zone) is compared against min_intact_cm:
      • span >= min_intact_cm → intact zones in the run stay green.
      • span <  min_intact_cm → intact zones in the run → fractured.
    Mechanical fracture zones always keep their "fractured" label (shown
    yellow for reference) regardless of the run outcome.
    """
    if not bypass_mechanical:
        # ── Standard per-zone check ───────────────────────────────────────────
        result = []
        for z in zones:
            if z["label"] == "intact":
                cs, ce = z.get("cm_start"), z.get("cm_end")
                length_cm = (ce - cs) if (cs is not None and ce is not None) \
                             else min_intact_cm   # unknown length → keep intact
                if length_cm < min_intact_cm:
                    z = {**z, "label": "fractured"}
            result.append(z)
        return result

    # ── Bypass mode: mechanical fractures are transparent for RQD ────────────
    result = [dict(z) for z in zones]

    def _is_mech(z):
        """Fractured zone whose origin is mechanical (p_mec > p_nat)."""
        return z["label"] == "fractured" and z.get("p_mec", 0) > z.get("p_nat", 0)

    def _is_run_zone(z):
        return z["label"] == "intact" or _is_mech(z)

    i = 0
    while i < len(result):
        if not _is_run_zone(result[i]):
            i += 1
            continue

        # Extend to find all consecutive intact / mechanical-fracture zones
        j = i
        while j < len(result) and _is_run_zone(result[j]):
            j += 1

        # Only act if at least one intact zone is in the run
        if any(z["label"] == "intact" for z in result[i:j]):
            cs = result[i].get("cm_start")
            ce = result[j - 1].get("cm_end")
            if cs is not None and ce is not None and (ce - cs) < min_intact_cm:
                # Run too short → demote intact zones (mechanical fractures stay)
                for k in range(i, j):
                    if result[k]["label"] == "intact":
                        result[k]["label"] = "fractured"

        i = j

    return result
