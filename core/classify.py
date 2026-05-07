DEFAULT_THRESHOLDS = {
    # ── Intact ────────────────────────────────────────────────────────────────
    "intact_continuity_min":    0.65,
    "intact_occupancy_min":     0.55,
    "intact_fragmentation_max": 0.25,
    "intact_components_max":    3,
    "intact_crack_max":         0.08,   # < 8 % dark columns → intact

    # ── Rubble ────────────────────────────────────────────────────────────────
    "rubble_fragmentation_min": 0.55,
    "rubble_occupancy_max":     0.40,
    "rubble_components_min":    6,
    "rubble_crack_min":         0.25,   # > 25 % dark columns → rubble

    # ── Fractured (middle band) ───────────────────────────────────────────────
    # Any crack_column_fraction between intact_crack_max and rubble_crack_min
    # lands here automatically.
}


def classify_windows(all_feats, thresholds=None):
    """
    Assign 'intact' | 'fractured' | 'rubble' to each window.
    Returns list of label strings.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    return [_classify_one(f, thr) for f in all_feats]


def _classify_one(f, thr):
    occ      = f["rock_occupancy"]
    n_comp   = f["connected_components_count"]
    lr       = f["largest_component_ratio"]
    cont     = f.get("longitudinal_continuity", 0.5)
    frag     = f["local_fragmentation"]
    ws       = f["width_stability"]
    crack_f  = f.get("crack_column_fraction", 0.0)
    bri_cv   = f.get("brightness_cv", 0.0)

    # Gap / empty strip
    if occ < 0.08:
        return "rubble"

    # ── Crack-based fast path ─────────────────────────────────────────────────
    # When the rock fills the frame, these signals are the most reliable.
    if crack_f >= thr["rubble_crack_min"]:
        return "rubble"

    if crack_f <= thr["intact_crack_max"]:
        # Very few columns with through-going dark coverage → intact candidate.
        # bri_cv excluded: granitic rock has inherently high brightness variance
        # (mixed minerals) regardless of integrity.
        if occ >= thr["intact_occupancy_min"] and frag <= thr["intact_fragmentation_max"]:
            return "intact"

    # ── Mass-based path (for fragmented / rubble-pile images) ─────────────────
    continuity_score = (
        0.30 * cont  +
        0.25 * lr    +
        0.25 * occ   +
        0.20 * ws
    )

    if (
        continuity_score >= thr["intact_continuity_min"] and
        occ              >= thr["intact_occupancy_min"]   and
        frag             <= thr["intact_fragmentation_max"] and
        n_comp           <= thr["intact_components_max"]   and
        crack_f          <= thr["intact_crack_max"]
    ):
        return "intact"

    if (
        frag   >= thr["rubble_fragmentation_min"] or
        occ    <= thr["rubble_occupancy_max"]      or
        n_comp >= thr["rubble_components_min"]
    ):
        return "rubble"

    return "fractured"
