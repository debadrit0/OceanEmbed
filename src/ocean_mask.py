"""
OceanEmbed land/ocean validation.

This module provides a deterministic land-mask check for user-selected
latitude/longitude locations.

The land/ocean mask is derived from the validated GLORYS surface
temperature field. A grid cell is considered ocean if at least one
valid GLORYS surface-temperature observation exists there during the
available time period.

This check is performed BEFORE model inference.

Returned result example:

{
    "is_ocean": False,
    "status": "LAND",
    "message": "Selected location is a land mass. Please select an ocean location.",
    "latitude": 22.5,
    "longitude": 88.25
}
"""

from pathlib import Path

import numpy as np
import xarray as xr

from .config import (
    LATS,
    LONS,
)


DEFAULT_GLORYS_PATH = (
    Path("data")
    / "processed"
    / "glorys_full.nc"
)


# Cached mask so we do not reopen/recalculate it for every user query.
_OCEAN_MASK = None


def _build_ocean_mask(
    glorys_path=DEFAULT_GLORYS_PATH,
):
    """
    Build a static ocean mask from GLORYS surface temperature.

    A cell is ocean when the 0 m temperature field contains at least
    one valid observation anywhere in the available time period.
    """

    glorys_path = Path(glorys_path)

    if not glorys_path.exists():
        raise FileNotFoundError(
            f"GLORYS file not found: {glorys_path}"
        )

    print(
        "Building OceanEmbed land/ocean mask..."
    )

    ds = xr.open_dataset(
        glorys_path
    )

    if "temperature" not in ds:
        ds.close()

        raise ValueError(
            "GLORYS dataset does not contain "
            "the required 'temperature' variable."
        )

    temperature = ds[
        "temperature"
    ]

    if "depth" not in temperature.dims:
        ds.close()

        raise ValueError(
            "GLORYS temperature has no depth dimension."
        )

    # Select the requested surface layer.
    surface = temperature.isel(
        depth=0
    )

    if "time" in surface.dims:
        mask = (
            surface
            .notnull()
            .any(dim="time")
        )
    else:
        mask = surface.notnull()

    mask = mask.transpose(
        "lat",
        "lon"
    )

    values = mask.values.astype(
        bool
    )

    ds.close()

    expected_shape = (
        len(LATS),
        len(LONS),
    )

    if values.shape != expected_shape:
        raise ValueError(
            f"Ocean mask has shape {values.shape}, "
            f"expected {expected_shape}."
        )

    print(
        "Ocean mask built:",
        values.shape
    )

    print(
        "Ocean cells:",
        int(values.sum())
    )

    print(
        "Land cells:",
        int((~values).sum())
    )

    print(
        "Ocean coverage: {:.2f}%".format(
            values.mean() * 100.0
        )
    )

    return values


def get_ocean_mask(
    glorys_path=DEFAULT_GLORYS_PATH,
):
    """
    Return the cached ocean mask.
    """

    global _OCEAN_MASK

    if _OCEAN_MASK is None:
        _OCEAN_MASK = _build_ocean_mask(
            glorys_path
        )

    return _OCEAN_MASK


def _nearest_grid_index(
    latitude,
    longitude,
):
    """
    Find the nearest OceanEmbed grid cell.
    """

    lat_index = int(
        np.argmin(
            np.abs(
                np.asarray(LATS)
                - latitude
            )
        )
    )

    lon_index = int(
        np.argmin(
            np.abs(
                np.asarray(LONS)
                - longitude
            )
        )
    )

    return lat_index, lon_index


def check_location(
    latitude,
    longitude,
    glorys_path=DEFAULT_GLORYS_PATH,
):
    """
    Check whether a user-selected location is ocean or land.

    Returns a dictionary suitable for an API/GUI response.
    """

    latitude = float(latitude)
    longitude = float(longitude)

    # -------------------------------------------------------------
    # Domain check
    # -------------------------------------------------------------

    if (
        latitude < float(LATS.min())
        or latitude > float(LATS.max())
        or longitude < float(LONS.min())
        or longitude > float(LONS.max())
    ):
        return {
            "is_ocean": False,
            "status": "OUT_OF_DOMAIN",
            "message": (
                "Selected location is outside the "
                "OceanEmbed North Indian Ocean domain."
            ),
            "latitude": latitude,
            "longitude": longitude,
            "nearest_grid_latitude": None,
            "nearest_grid_longitude": None,
        }

    # -------------------------------------------------------------
    # Get mask
    # -------------------------------------------------------------

    mask = get_ocean_mask(
        glorys_path
    )

    lat_index, lon_index = (
        _nearest_grid_index(
            latitude,
            longitude
        )
    )

    is_ocean = bool(
        mask[
            lat_index,
            lon_index
        ]
    )

    nearest_lat = float(
        LATS[lat_index]
    )

    nearest_lon = float(
        LONS[lon_index]
    )

    # -------------------------------------------------------------
    # User-facing response
    # -------------------------------------------------------------

    if is_ocean:
        return {
            "is_ocean": True,
            "status": "OCEAN",
            "message": (
                "Selected location is over the ocean. "
                "OceanEmbed inference can proceed."
            ),
            "latitude": latitude,
            "longitude": longitude,
            "nearest_grid_latitude": nearest_lat,
            "nearest_grid_longitude": nearest_lon,
        }

    return {
        "is_ocean": False,
        "status": "LAND",
        "message": (
            "Selected location is a land mass. "
            "Please select a location over the ocean."
        ),
        "latitude": latitude,
        "longitude": longitude,
        "nearest_grid_latitude": nearest_lat,
        "nearest_grid_longitude": nearest_lon,
    }


def assert_ocean_location(
    latitude,
    longitude,
    glorys_path=DEFAULT_GLORYS_PATH,
):
    """
    Validate that a location is ocean.

    Raises ValueError for land or out-of-domain locations.

    This is useful immediately before model inference.
    """

    result = check_location(
        latitude,
        longitude,
        glorys_path
    )

    if not result["is_ocean"]:
        raise ValueError(
            result["message"]
        )

    return result


def main():
    """
    Standalone test.
    """

    print(
        "=" * 80
    )

    print(
        "OCEANEMBED LAND/OCEAN MASK TEST"
    )

    print(
        "=" * 80
    )

    mask = get_ocean_mask()

    print()
    print(
        "Mask shape:",
        mask.shape
    )

    # A few representative checks.
    test_locations = [
        (20.0, 70.0),
        (15.0, 80.0),
        (25.0, 85.0),
        (22.5, 88.25),
    ]

    print()
    print(
        "Location tests:"
    )

    for latitude, longitude in test_locations:

        result = check_location(
            latitude,
            longitude
        )

        print()
        print(
            f"({latitude}, {longitude})"
        )

        print(
            "Status:",
            result["status"]
        )

        print(
            "Message:",
            result["message"]
        )

        print(
            "Nearest grid:",
            (
                result[
                    "nearest_grid_latitude"
                ],
                result[
                    "nearest_grid_longitude"
                ],
            )
        )

    print()
    print(
        "=" * 80
    )

    print(
        "LAND/OCEAN MASK TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
    