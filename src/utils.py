import numpy as np
import torch
import pyro
from typing import List

# FIPS county prefixes (5-digit strings)
CITY_TO_COUNTY_PREFIXES = {
    "durham": ["37063"],
    "raleigh": ["37183"],
    "charlotte": ["37119"],
    "asheville": ["37021"]
}


def county_prefixes_for_city(city_name: str) -> List[str]:
    """
    Return list of county FIPS prefixes for a given city.

    Raises a clear error if the city is unknown.
    """
    key = city_name.lower()

    if key not in CITY_TO_COUNTY_PREFIXES:
        known = ", ".join(sorted(CITY_TO_COUNTY_PREFIXES.keys()))
        raise ValueError(
            f"Unknown city_name='{city_name}'. "
            f"Known cities: {known}. "
            f"Please add it to CITY_TO_COUNTY_PREFIXES."
        )

    return CITY_TO_COUNTY_PREFIXES[key]


def set_seeds(seed: int, torch_dtype: str = "float32") -> torch.dtype:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if torch_dtype == "float32":
        dtype = torch.float32
    elif torch_dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype} (use float32 or float64)")
    torch.set_default_dtype(dtype)
    return dtype


def t_s_col_from_slot(time_slot: str) -> str:
    if time_slot == "eve":
        return "t_s_night"
    if time_slot in ("morn", "af"):
        return "t_s_day"
    raise ValueError("Invalid time_slot. Choose 'morn', 'af', or 'eve'.")


def abbrv_time_slot(time_slot: str) -> str:
    if time_slot == "eve":
        return "pm"
    if time_slot == "morn":
        return "am"
    if time_slot == "af":
        return "af"
    raise ValueError("Invalid time_slot. Choose 'morn', 'af', or 'eve'.")


def census_abbr(census_group_type: str) -> str:
    return "tract" if census_group_type == "tract" else "bg"

def geoseries_union_all(gs):
    """
    Robust union across geopandas/shapely versions.
    gs: GeoSeries
    """
    if hasattr(gs, "union_all"):
        return gs.union_all()

    if hasattr(gs, "unary_union"):
        return gs.unary_union

    from shapely.ops import unary_union
    return unary_union(list(gs.values))
