"""
REQ-6: hard filters (FROZEN v1.0).

Non-negotiable constraints, applied as vectorized masks over the whole corpus
**before** retrieval (D2). A job failing any filter is eliminated, not penalized:
giving salary a 5% weight would still let an underpaying job rank first, whereas
a filter will not.

Running before retrieval is the correction to the draft plan's ordering. For a
narrow profile roughly 1-3% of a national corpus survives, so retrieving first
would have left a handful of candidates to rank.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import pandas as pd

from src.profiles import UserProfile
from src.seq import as_list, as_set

#: AC-6.3: qualifiers stripped before comparison, so "Kansas City Metropolitan
#: Area" and "Kansas City, MO" reach the same normalized form.
_QUALIFIERS = re.compile(
    r"\b(metropolitan|metro|greater|area|region|county|and vicinity|united states|usa)\b", re.I
)
_WS = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _states() -> tuple[dict[str, str], set[str]]:
    import geonamescache

    gc = geonamescache.GeonamesCache()
    by_name = {v["name"].lower(): k for k, v in gc.get_us_states().items()}
    return by_name, set(gc.get_us_states())


STATE_TO_CODE: dict[str, str] = _states()[0]


@lru_cache(maxsize=1)
def _cities() -> dict[tuple[str, str], tuple[float, float]]:
    """(city, state) -> (lat, lon), keeping the most populous on collision."""
    import geonamescache

    out: dict[tuple[str, str], tuple[float, float]] = {}
    best: dict[tuple[str, str], int] = {}
    for c in geonamescache.GeonamesCache().get_cities().values():
        if c["countrycode"] != "US":
            continue
        key = (c["name"].lower(), c["admin1code"])
        if c["population"] >= best.get(key, -1):
            best[key] = c["population"]
            out[key] = (c["latitude"], c["longitude"])
    return out


def normalize_location(raw: str | None) -> str:
    """AC-6.3: lowercase, strip qualifiers and punctuation, collapse whitespace."""
    if not raw:
        return ""
    return _WS.sub(" ", _QUALIFIERS.sub(" ", raw.lower()).replace(",", " ")).strip()


def parse_location(raw: str | None) -> tuple[str | None, str | None]:
    """
    AC-6.3: ``(city, state_code)``. State names are expanded to codes
    (Missouri -> MO); metro/greater/area qualifiers are stripped.
    """
    if not raw:
        return None, None
    by_name, codes = _states()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    state = None
    if len(parts) >= 2:
        tail = _QUALIFIERS.sub(" ", parts[-1]).strip()
        if tail.upper() in codes:
            state = tail.upper()
        elif tail.lower() in by_name:
            state = by_name[tail.lower()]
    city = normalize_location(parts[0]) or None
    return city, state


def geocode(city: str | None, state: str | None) -> tuple[float, float] | None:
    """
    Offline lookup via `geonamescache` — no API key, no rate limit, no runtime
    download. Covers US cities above ~15k population (3,272 of them), so real
    suburbs like Overland Park resolve while small exurbs fall to AC-6.2's
    string-equality branch.
    """
    if not city:
        return None
    cities = _cities()
    if state:
        return cities.get((city, state))
    matches = [(k, v) for k, v in cities.items() if k[0] == city]
    return matches[0][1] if len(matches) == 1 else None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


# --- individual filters -----------------------------------------------------

def location_passes(job: pd.Series, profile: UserProfile) -> bool:
    """AC-6.2's five-branch decision tree, in order."""
    # (a) remote-only user
    if profile.accepted_work_settings == {"Remote"}:
        return bool(job.get("is_remote"))
    # (b) remote job — geography does not apply
    if job.get("is_remote"):
        return True
    if not profile.preferred_location:
        return True

    j_city, j_state = parse_location(job.get("location_raw"))
    u_city, u_state = parse_location(profile.preferred_location)
    j_pt, u_pt = geocode(j_city, j_state), geocode(u_city, u_state)

    # (c) both resolved — distance decides
    if j_pt and u_pt:
        return haversine_miles(*u_pt, *j_pt) <= profile.max_distance_miles
    # (d) either unresolved — normalized string equality
    if normalize_location(job.get("location_raw")) == normalize_location(profile.preferred_location):
        return True
    # (e) unresolved and unequal
    return False


def salary_passes(job: pd.Series, profile: UserProfile) -> bool:
    """
    AC-6.5. Unlisted salary passes **iff** the user opted in, which makes the
    missing-data policy a visible choice rather than a hidden rule. Measured:
    only 30.1% of the corpus lists a salary, so this branch carries most rows.
    """
    if not profile.min_salary:
        return True
    if not job.get("salary_listed"):
        return bool(profile.include_unlisted_salary)
    top = job.get("salary_max")
    return top is not None and top == top and top >= profile.min_salary


def work_setting_passes(job: pd.Series, profile: UserProfile) -> bool:
    """AC-6.6: set membership, not equality (D5). Unknown passes and is tagged."""
    value = job.get("work_setting")
    return value == "Unknown" or value in profile.accepted_work_settings


def employment_type_passes(job: pd.Series, profile: UserProfile) -> bool:
    """AC-6.7: same policy as work setting."""
    value = job.get("employment_type")
    return value == "Unknown" or value in profile.accepted_employment_types


def skill_floor_passes(job: pd.Series, profile: UserProfile) -> bool:
    """AC-6.8: at least one shared required skill."""
    return bool(as_set(job.get("required_skills")) & profile.skills)


# --- vectorized masks (AC-6.1) ----------------------------------------------
# The per-row predicates above are the readable, individually-tested definition;
# these produce the same answers as boolean masks over the whole frame. D2's
# argument for filtering before retrieval is that this stage is cheap, so it has
# to actually be cheap - row-wise .apply() over 19k rows is not.
# `test_vectorized_matches_rowwise` asserts the two paths agree.


def _location_mask(jobs: pd.DataFrame, profile: UserProfile) -> pd.Series:
    """
    Geocoding is resolved once per *distinct* location, not once per row - a
    19k-row corpus holds only a few thousand distinct locations.
    """
    if profile.accepted_work_settings == {"Remote"}:
        return jobs["is_remote"].fillna(False).astype(bool)

    remote = jobs["is_remote"].fillna(False).astype(bool)
    if not profile.preferred_location:
        return pd.Series(True, index=jobs.index)

    u_city, u_state = parse_location(profile.preferred_location)
    u_pt = geocode(u_city, u_state)
    u_norm = normalize_location(profile.preferred_location)

    def resolve(raw) -> bool:
        j_city, j_state = parse_location(raw)
        j_pt = geocode(j_city, j_state)
        if j_pt and u_pt:
            return haversine_miles(*u_pt, *j_pt) <= profile.max_distance_miles
        return normalize_location(raw) == u_norm

    decisions = {loc: resolve(loc) for loc in jobs["location_raw"].dropna().unique()}
    near = jobs["location_raw"].map(decisions).fillna(False).astype(bool)
    return remote | near


def _salary_mask(jobs: pd.DataFrame, profile: UserProfile) -> pd.Series:
    if not profile.min_salary:
        return pd.Series(True, index=jobs.index)
    listed = jobs["salary_listed"].fillna(False).astype(bool)
    high_enough = jobs["salary_max"].fillna(-1) >= profile.min_salary
    return (listed & high_enough) | (~listed & bool(profile.include_unlisted_salary))


def _skill_floor_mask(jobs: pd.DataFrame, profile: UserProfile) -> pd.Series:
    """
    Note `as_set`: Parquet round-trips list columns as numpy arrays, where
    `v or []` raises rather than returning a default.
    """
    skills = profile.skills
    return pd.Series(
        [bool(as_set(v) & skills) for v in jobs["required_skills"]], index=jobs.index
    )


MASKS = [
    ("location", _location_mask),
    ("salary", _salary_mask),
    ("work_setting", lambda j, p: j["work_setting"].eq("Unknown")
        | j["work_setting"].isin(p.accepted_work_settings)),
    ("employment_type", lambda j, p: j["employment_type"].eq("Unknown")
        | j["employment_type"].isin(p.accepted_employment_types)),
    ("skill_floor", _skill_floor_mask),
]

FILTERS = [
    ("location", location_passes),
    ("salary", salary_passes),
    ("work_setting", work_setting_passes),
    ("employment_type", employment_type_passes),
    ("skill_floor", skill_floor_passes),
]

_SUGGESTIONS = {
    "location": "widen the distance, or accept remote roles",
    "salary": "lower the minimum salary, or include jobs with no listed salary",
    "work_setting": "accept more work settings",
    "employment_type": "accept more employment types",
    "skill_floor": "add skills to your profile",
}


def warm_caches() -> None:
    """
    Build the city and state lookups up front. Without this the first call pays
    a one-off ~300ms to construct them, which would put a cold first search over
    AC-6.1's 200ms budget while every later search runs in 10-70ms. Called at
    app start.
    """
    _cities()
    _states()


def apply_filters(jobs: pd.DataFrame, profile: UserProfile) -> tuple[pd.DataFrame, dict]:
    """
    AC-6.9: compose with AND and return the survivor funnel.

    AC-6.10: when nothing survives, name the filter that eliminated the most and
    suggest relaxing it — never silently return zero results.
    """
    funnel: dict = {"start": len(jobs)}
    eliminated: dict[str, int] = {}
    kept = jobs

    for name, mask_fn in MASKS:
        before = len(kept)
        kept = kept[mask_fn(kept, profile)] if before else kept
        funnel[name] = len(kept)
        eliminated[name] = before - len(kept)

    funnel["eliminated"] = eliminated
    if len(kept) == 0 and len(jobs):
        worst = max(eliminated, key=eliminated.get)
        funnel["limiting_filter"] = worst
        funnel["suggestion"] = (
            f"No jobs matched. The {worst.replace('_', ' ')} filter removed the most "
            f"({eliminated[worst]}); try to {_SUGGESTIONS[worst]}."
        )
    return kept, funnel
