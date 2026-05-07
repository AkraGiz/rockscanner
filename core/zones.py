from core.smooth import get_zones


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


def filter_short_intact(zones, min_intact_cm):
    """
    Reclassify intact zones shorter than min_intact_cm as fractured.
    Zones without cm info use pct_start/pct_end (shouldn't happen with default 100 cm).
    Adjacent fractured zones that become neighbours are left as-is (smoothing
    would merge them, but here we keep it simple and readable).
    """
    result = []
    for z in zones:
        if z["label"] == "intact":
            cm_s = z.get("cm_start")
            cm_e = z.get("cm_end")
            if cm_s is not None and cm_e is not None:
                length_cm = cm_e - cm_s
            else:
                length_cm = min_intact_cm  # unknown length → keep intact
            if length_cm < min_intact_cm:
                z = {**z, "label": "fractured"}
        result.append(z)
    return result
