import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import copernicusmarine
import numpy as np


# ============================================================
# OCEANEMBED DOMAIN
# ============================================================

LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0

GRID_STEP = 0.25

TARGET_LAT = np.arange(
    LAT_MIN,
    LAT_MAX + 0.001,
    GRID_STEP,
    dtype=np.float32
)

TARGET_LON = np.arange(
    LON_MIN,
    LON_MAX + 0.001,
    GRID_STEP,
    dtype=np.float32
)

PATCH_SIZE = 9
HALF_PATCH = PATCH_SIZE // 2


# ============================================================
# DATASET IDS
# ============================================================

SST_DATASET = (
    "METOFFICE-GLO-SST-L4-REP-OBS-SST"
)

SSS_DATASET = (
    "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
)

SSH_DATASET = (
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
)

CURRENT_DATASET = (
    "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"
)

WIND_ASC_DATASET = (
    "cmems_obs-wind_glo_phy_my_l3-metopb-ascat-asc-0.25deg_P1D-i"
)

WIND_DES_DATASET = (
    "cmems_obs-wind_glo_phy_my_l3-metopb-ascat-des-0.25deg_P1D-i"
)


# ============================================================
# VARIABLE NAMES
# ============================================================

SST_VAR = "analysed_sst"
SSS_VAR = "sos"
SSH_VAR = "sla"

U_CURRENT_VAR = "uo"
V_CURRENT_VAR = "vo"

U_WIND_VAR = "eastward_wind"
V_WIND_VAR = "northward_wind"


# ============================================================
# SSS KNOWN DATA LIMIT
#
# The Copernicus SSS source used here has data through
# 2024-12-15 in the current validation run.
#
# For 2024-12-16 onward, we carry forward the 2024-12-15
# SSS field instead of inventing data.
# ============================================================

SSS_LAST_AVAILABLE = datetime.strptime(
    "2024-12-15",
    "%Y-%m-%d"
)


# ============================================================
# NORMALIZATION
# ============================================================

NORMALIZATION_MEAN = np.array(
    [
        27.0,
        35.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32
)

NORMALIZATION_STD = np.array(
    [
        5.0,
        2.0,
        0.5,
        0.5,
        0.5,
        8.0,
        8.0,
    ],
    dtype=np.float32
)


# ============================================================
# COORDINATE HELPERS
# ============================================================

def _rename_coordinates(ds):

    rename_map = {}

    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _find_variable(ds, preferred_name):

    if preferred_name in ds.data_vars:
        return preferred_name

    lower_map = {
        name.lower(): name
        for name in ds.data_vars
    }

    if preferred_name.lower() in lower_map:

        return lower_map[
            preferred_name.lower()
        ]

    raise KeyError(
        f"Variable '{preferred_name}' not found. "
        f"Available variables: "
        f"{list(ds.data_vars)}"
    )


def _prepare_dataarray(da):

    da = da.squeeze(drop=True)

    if "latitude" in da.dims:

        da = da.rename(
            {
                "latitude": "lat"
            }
        )

    if "longitude" in da.dims:

        da = da.rename(
            {
                "longitude": "lon"
            }
        )

    if "time" in da.dims:

        da = da.sortby("time")

    if "lat" in da.coords:

        da = da.sortby("lat")

    if "lon" in da.coords:

        da = da.sortby("lon")

    return da


# ============================================================
# SELECT SURFACE DEPTH
# ============================================================

def _select_surface_depth(
    da,
    variable_name
):

    if "depth" not in da.dims:

        return da

    print(
        f"        {variable_name} contains "
        f"depth dimension: "
        f"{da.sizes.get('depth')}"
    )

    if "depth" in da.coords:

        depth_values = np.asarray(
            da["depth"].values,
            dtype=np.float32
        )

        print(
            f"        available depths: "
            f"{depth_values}"
        )

        exact_zero = np.where(
            np.isclose(
                depth_values,
                0.0,
                atol=0.01
            )
        )[0]

        if len(exact_zero) > 0:

            da = da.isel(
                depth=int(
                    exact_zero[0]
                )
            )

        else:

            da = da.isel(
                depth=0
            )

    else:

        da = da.isel(
            depth=0
        )

    print(
        f"        {variable_name} after "
        f"surface-depth selection: "
        f"{da.dims}"
    )

    return da


# ============================================================
# FORCE EXACT BLOCK SHAPE
# ============================================================

def _force_block_shape(
    data,
    expected_days,
    variable_name
):

    data = np.asarray(
        data,
        dtype=np.float32
    )

    print(
        f"        {variable_name} raw shape: "
        f"{data.shape}"
    )

    data = np.squeeze(data)

    if data.ndim == 2:

        data = data[
            np.newaxis,
            ...
        ]

    if data.ndim != 3:

        raise RuntimeError(
            f"{variable_name} has unsupported "
            f"shape {data.shape}. "
            f"Expected [time, lat, lon]."
        )

    if data.shape[0] > expected_days:

        data = data[
            :expected_days
        ]

    elif data.shape[0] < expected_days:

        padded = np.full(
            (
                expected_days,
                data.shape[1],
                data.shape[2]
            ),
            np.nan,
            dtype=np.float32
        )

        padded[
            :data.shape[0]
        ] = data

        data = padded

    expected_lat = len(TARGET_LAT)
    expected_lon = len(TARGET_LON)

    if (
        data.shape[1] != expected_lat
        or
        data.shape[2] != expected_lon
    ):

        raise RuntimeError(
            f"{variable_name} spatial shape "
            f"{data.shape[1:]} does not match "
            f"expected "
            f"({expected_lat}, "
            f"{expected_lon})"
        )

    print(
        f"        {variable_name} final shape: "
        f"{data.shape}"
    )

    return data.astype(
        np.float32
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_surface(surface):

    result = (
        surface
        .astype(np.float32)
        .copy()
    )

    for channel in range(7):

        result[:, channel] = (
            result[:, channel]
            - NORMALIZATION_MEAN[channel]
        ) / NORMALIZATION_STD[channel]

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return result


# ============================================================
# OPEN GENERAL VARIABLE
# ============================================================

def _open_variable(
    dataset_id,
    variable,
    start_date,
    end_date,
    expected_days,
    convert_kelvin=False
):

    print()
    print(
        f"      opening {dataset_id}"
    )

    print(
        f"      variable: {variable}"
    )

    print(
        f"      dates: "
        f"{start_date} -> {end_date}"
    )

    ds = copernicusmarine.open_dataset(

        dataset_id=dataset_id,

        variables=[variable],

        minimum_longitude=LON_MIN,

        maximum_longitude=LON_MAX,

        minimum_latitude=LAT_MIN,

        maximum_latitude=LAT_MAX,

        start_datetime=start_date,

        end_datetime=end_date
    )

    ds = _rename_coordinates(ds)

    variable_name = _find_variable(
        ds,
        variable
    )

    da = _prepare_dataarray(
        ds[variable_name]
    )

    da = _select_surface_depth(
        da,
        variable
    )

    if "time" not in da.dims:

        da = da.expand_dims(
            time=[
                np.datetime64(start_date)
            ]
        )

    da = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear"
    )

    unwanted_dims = [
        d
        for d in da.dims
        if d not in (
            "time",
            "lat",
            "lon"
        )
    ]

    for dim in unwanted_dims:

        da = da.isel(
            {
                dim: 0
            }
        )

    da = da.transpose(
        "time",
        "lat",
        "lon"
    )

    data = da.values.astype(
        np.float32
    )

    if convert_kelvin:

        data = (
            data - 273.15
        )

    data = _force_block_shape(
        data,
        expected_days,
        variable
    )

    try:
        ds.close()
    except Exception:
        pass

    return data


# ============================================================
# SPECIAL SSS LOADER
# ============================================================

def _open_sss(
    start_date,
    end_date,
    expected_days
):

    requested_start = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    requested_end = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # If the entire request is within available SSS data,
    # use the normal loader.
    # --------------------------------------------------------

    if requested_end <= SSS_LAST_AVAILABLE:

        return _open_variable(

            SSS_DATASET,

            SSS_VAR,

            start_date,

            end_date,

            expected_days

        ), False


    # --------------------------------------------------------
    # Request begins after the available SSS period.
    # This should only happen for the final 2024 dates.
    # --------------------------------------------------------

    if requested_start > SSS_LAST_AVAILABLE:

        reference_date = (
            SSS_LAST_AVAILABLE.strftime(
                "%Y-%m-%d"
            )
        )

        print()
        print(
            "WARNING: Requested SSS period "
            f"{start_date} -> {end_date} "
            "is after the available SSS data."
        )

        print(
            "Using SSS from "
            f"{reference_date} "
            "for this entire block."
        )

        reference = _open_variable(

            SSS_DATASET,

            SSS_VAR,

            reference_date,

            reference_date,

            1

        )

        result = np.repeat(

            reference,

            expected_days,

            axis=0

        )

        return result, True


    # --------------------------------------------------------
    # Request overlaps the SSS limit.
    #
    # Example:
    # 2024-11-26 -> 2024-12-25
    #
    # Load available dates first, then carry forward the
    # last available SSS observation for missing days.
    # --------------------------------------------------------

    available_end = (
        SSS_LAST_AVAILABLE.strftime(
            "%Y-%m-%d"
        )
    )

    available_days = (
        SSS_LAST_AVAILABLE
        - requested_start
    ).days + 1

    print()
    print(
        "WARNING: SSS source ends at "
        f"{available_end}."
    )

    print(
        f"Loading SSS only through "
        f"{available_end}."
    )

    available_data = _open_variable(

        SSS_DATASET,

        SSS_VAR,

        start_date,

        available_end,

        available_days

    )

    missing_days = (
        expected_days
        - available_data.shape[0]
    )

    if missing_days <= 0:

        return available_data, False


    print(
        f"SSS missing for "
        f"{missing_days} day(s)."
    )

    print(
        "Carrying forward the last "
        "available SSS field."
    )

    last_field = (
        available_data[-1:]
    )

    repeated = np.repeat(
        last_field,
        missing_days,
        axis=0
    )

    result = np.concatenate(
        [
            available_data,
            repeated
        ],
        axis=0
    )

    return result, True


# ============================================================
# WIND
# ============================================================

def _open_wind(
    dataset_id,
    start_date,
    end_date,
    expected_days
):

    print()
    print(
        f"      opening wind: "
        f"{dataset_id}"
    )

    print(
        f"      dates: "
        f"{start_date} -> "
        f"{end_date}"
    )

    ds = copernicusmarine.open_dataset(

        dataset_id=dataset_id,

        variables=[
            U_WIND_VAR,
            V_WIND_VAR
        ],

        minimum_longitude=LON_MIN,

        maximum_longitude=LON_MAX,

        minimum_latitude=LAT_MIN,

        maximum_latitude=LAT_MAX,

        start_datetime=start_date,

        end_datetime=end_date

    )

    ds = _rename_coordinates(
        ds
    )

    u_name = _find_variable(
        ds,
        U_WIND_VAR
    )

    v_name = _find_variable(
        ds,
        V_WIND_VAR
    )

    u = _prepare_dataarray(
        ds[u_name]
    )

    v = _prepare_dataarray(
        ds[v_name]
    )

    u = _select_surface_depth(
        u,
        U_WIND_VAR
    )

    v = _select_surface_depth(
        v,
        V_WIND_VAR
    )

    if "time" not in u.dims:

        u = u.expand_dims(
            time=[
                np.datetime64(
                    start_date
                )
            ]
        )

    if "time" not in v.dims:

        v = v.expand_dims(
            time=[
                np.datetime64(
                    start_date
                )
            ]
        )

    u = u.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear"
    )

    v = v.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear"
    )

    for dim in list(u.dims):

        if dim not in (
            "time",
            "lat",
            "lon"
        ):

            u = u.isel(
                {
                    dim: 0
                }
            )

    for dim in list(v.dims):

        if dim not in (
            "time",
            "lat",
            "lon"
        ):

            v = v.isel(
                {
                    dim: 0
                }
            )

    u = u.transpose(
        "time",
        "lat",
        "lon"
    )

    v = v.transpose(
        "time",
        "lat",
        "lon"
    )

    u_data = _force_block_shape(
        u.values,
        expected_days,
        "u_wind"
    )

    v_data = _force_block_shape(
        v.values,
        expected_days,
        "v_wind"
    )

    try:
        ds.close()
    except Exception:
        pass

    return (
        u_data,
        v_data
    )


# ============================================================
# FULL ONLINE BLOCK
# ============================================================

def load_surface_block(
    start_date,
    end_date
):

    start_dt = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    end_dt = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    )

    dates = []

    current = start_dt

    while current <= end_dt:

        dates.append(
            current.strftime(
                "%Y-%m-%d"
            )
        )

        current += timedelta(
            days=1
        )

    expected_days = len(
        dates
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ONLINE SURFACE BLOCK"
    )

    print(
        f"{start_date} -> {end_date}"
    )

    print(
        "Expected days:",
        expected_days
    )

    print(
        f"Target grid: "
        f"{len(TARGET_LAT)} x "
        f"{len(TARGET_LON)}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # SST
    # --------------------------------------------------------

    sst = _open_variable(

        SST_DATASET,

        SST_VAR,

        start_date,

        end_date,

        expected_days,

        convert_kelvin=True

    )


    # --------------------------------------------------------
    # SSS
    # --------------------------------------------------------

    sss, sss_carried_forward = _open_sss(

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # SSH
    # --------------------------------------------------------

    ssh = _open_variable(

        SSH_DATASET,

        SSH_VAR,

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # U CURRENT
    # --------------------------------------------------------

    u_current = _open_variable(

        CURRENT_DATASET,

        U_CURRENT_VAR,

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # V CURRENT
    # --------------------------------------------------------

    v_current = _open_variable(

        CURRENT_DATASET,

        V_CURRENT_VAR,

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # ASCENDING WIND
    # --------------------------------------------------------

    u_wind_a, v_wind_a = _open_wind(

        WIND_ASC_DATASET,

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # DESCENDING WIND
    # --------------------------------------------------------

    u_wind_d, v_wind_d = _open_wind(

        WIND_DES_DATASET,

        start_date,

        end_date,

        expected_days

    )


    # --------------------------------------------------------
    # WIND AVERAGE
    # --------------------------------------------------------

    u_wind = np.nanmean(

        np.stack(
            [
                u_wind_a,
                u_wind_d
            ],
            axis=0
        ),

        axis=0

    )


    v_wind = np.nanmean(

        np.stack(
            [
                v_wind_a,
                v_wind_d
            ],
            axis=0
        ),

        axis=0

    )


    # --------------------------------------------------------
    # FINAL SHAPE CHECK
    # --------------------------------------------------------

    arrays = {

        "SST": sst,

        "SSS": sss,

        "SSH": ssh,

        "U CURRENT": u_current,

        "V CURRENT": v_current,

        "U WIND": u_wind,

        "V WIND": v_wind

    }


    expected_shape = (

        expected_days,

        len(TARGET_LAT),

        len(TARGET_LON)

    )


    print()
    print(
        "FINAL INPUT SHAPES:"
    )


    for name, array in arrays.items():

        print(
            f"  {name:12s}: "
            f"{array.shape}"
        )

        if array.shape != expected_shape:

            raise RuntimeError(

                f"{name} has shape "
                f"{array.shape}. "

                f"Expected "
                f"{expected_shape}."

            )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    surface_raw = np.stack(

        [
            sst,
            sss,
            ssh,
            u_current,
            v_current,
            u_wind,
            v_wind
        ],

        axis=1

    ).astype(
        np.float32
    )


    print()
    print(
        "Surface raw shape:",
        surface_raw.shape
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    surface_norm = normalize_surface(
        surface_raw
    )


    print(
        "Surface normalized shape:",
        surface_norm.shape
    )


    if sss_carried_forward:

        print()
        print(
            "NOTE:"
        )

        print(
            "SSS was unavailable for part of "
            "this requested period."
        )

        print(
            "The last available SSS field "
            "was carried forward."
        )


    print(
        "=" * 70
    )


    return (
        surface_norm,
        surface_raw,
        dates
    )


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patch(
    surface_day,
    center_i,
    center_j
):

    i0 = (
        center_i
        - HALF_PATCH
    )

    i1 = (
        center_i
        + HALF_PATCH
        + 1
    )

    j0 = (
        center_j
        - HALF_PATCH
    )

    j1 = (
        center_j
        + HALF_PATCH
        + 1
    )

    if (
        i0 < 0
        or j0 < 0
        or i1 > len(TARGET_LAT)
        or j1 > len(TARGET_LON)
    ):

        return None

    patch = surface_day[
        :,
        i0:i1,
        j0:j1
    ]

    if patch.shape != (
        7,
        PATCH_SIZE,
        PATCH_SIZE
    ):

        return None

    return patch


# ============================================================
# NEAREST GRID POINT
# ============================================================

def nearest_grid_point(
    latitude,
    longitude
):

    i = int(
        np.abs(
            TARGET_LAT - latitude
        ).argmin()
    )

    j = int(
        np.abs(
            TARGET_LON - longitude
        ).argmin()
    )

    return i, j


# ============================================================
# LOCAL GRID
# ============================================================

def _local_grid(
    latitude,
    longitude
):

    local_lat = (

        latitude

        + np.arange(
            -HALF_PATCH,
            HALF_PATCH + 1
        )

        * GRID_STEP

    )

    local_lon = (

        longitude

        + np.arange(
            -HALF_PATCH,
            HALF_PATCH + 1
        )

        * GRID_STEP

    )

    return (
        local_lat,
        local_lon
    )


# ============================================================
# OPEN LOCAL DATASET
# ============================================================

def _open_local(

    dataset_id,

    variables,

    date_text,

    local_lat,

    local_lon

):

    ds = copernicusmarine.open_dataset(

        dataset_id=dataset_id,

        variables=variables,

        minimum_longitude=float(
            np.min(local_lon)
        ),

        maximum_longitude=float(
            np.max(local_lon)
        ),

        minimum_latitude=float(
            np.min(local_lat)
        ),

        maximum_latitude=float(
            np.max(local_lat)
        ),

        start_datetime=date_text,

        end_datetime=date_text

    )

    ds = _rename_coordinates(
        ds
    )

    return ds


# ============================================================
# LOCAL VARIABLE
# ============================================================

def _local_variable(

    ds,

    variable,

    local_lat,

    local_lon

):

    variable_name = _find_variable(
        ds,
        variable
    )

    da = _prepare_dataarray(
        ds[variable_name]
    )

    da = _select_surface_depth(
        da,
        variable
    )

    if "time" in da.dims:

        da = da.isel(
            time=0
        )

    for dim in list(da.dims):

        if dim not in (
            "lat",
            "lon"
        ):

            da = da.isel(
                {
                    dim: 0
                }
            )

    da = da.interp(

        lat=local_lat,

        lon=local_lon,

        method="linear"

    )

    data = da.values.astype(
        np.float32
    )

    data = np.squeeze(
        data
    )

    if data.ndim != 2:

        raise RuntimeError(

            f"{variable} local shape "
            f"is {data.shape}"

        )

    if variable == SST_VAR:

        data = (
            data - 273.15
        )

    return data


# ============================================================
# FINAL LOCAL DEMO PATCH
# ============================================================

def load_surface_patch(

    date_text,

    latitude,

    longitude

):

    local_lat, local_lon = (
        _local_grid(
            latitude,
            longitude
        )
    )

    print()
    print(
        "Fetching online 9x9 "
        "surface patch..."
    )


    # SST

    ds = _open_local(

        SST_DATASET,

        [SST_VAR],

        date_text,

        local_lat,

        local_lon

    )

    sst = _local_variable(

        ds,

        SST_VAR,

        local_lat,

        local_lon

    )

    ds.close()


    # SSS

    requested_date = datetime.strptime(
        date_text,
        "%Y-%m-%d"
    )


    if requested_date <= SSS_LAST_AVAILABLE:

        ds = _open_local(

            SSS_DATASET,

            [SSS_VAR],

            date_text,

            local_lat,

            local_lon

        )

        sss = _local_variable(

            ds,

            SSS_VAR,

            local_lat,

            local_lon

        )

        ds.close()

    else:

        print(
            "WARNING: SSS unavailable for "
            f"{date_text}."
        )

        print(
            "Using SSS from "
            "2024-12-15."
        )

        ds = _open_local(

            SSS_DATASET,

            [SSS_VAR],

            "2024-12-15",

            local_lat,

            local_lon

        )

        sss = _local_variable(

            ds,

            SSS_VAR,

            local_lat,

            local_lon

        )

        ds.close()


    # SSH

    ds = _open_local(

        SSH_DATASET,

        [SSH_VAR],

        date_text,

        local_lat,

        local_lon

    )

    ssh = _local_variable(

        ds,

        SSH_VAR,

        local_lat,

        local_lon

    )

    ds.close()


    # CURRENTS

    ds = _open_local(

        CURRENT_DATASET,

        [
            U_CURRENT_VAR,
            V_CURRENT_VAR
        ],

        date_text,

        local_lat,

        local_lon

    )

    u_current = _local_variable(

        ds,

        U_CURRENT_VAR,

        local_lat,

        local_lon

    )

    v_current = _local_variable(

        ds,

        V_CURRENT_VAR,

        local_lat,

        local_lon

    )

    ds.close()


    # ASCENDING WIND

    ds = _open_local(

        WIND_ASC_DATASET,

        [
            U_WIND_VAR,
            V_WIND_VAR
        ],

        date_text,

        local_lat,

        local_lon

    )

    u_wind_a = _local_variable(

        ds,

        U_WIND_VAR,

        local_lat,

        local_lon

    )

    v_wind_a = _local_variable(

        ds,

        V_WIND_VAR,

        local_lat,

        local_lon

    )

    ds.close()


    # DESCENDING WIND

    ds = _open_local(

        WIND_DES_DATASET,

        [
            U_WIND_VAR,
            V_WIND_VAR
        ],

        date_text,

        local_lat,

        local_lon

    )

    u_wind_d = _local_variable(

        ds,

        U_WIND_VAR,

        local_lat,

        local_lon

    )

    v_wind_d = _local_variable(

        ds,

        V_WIND_VAR,

        local_lat,

        local_lon

    )

    ds.close()


    # Wind average

    u_wind = np.nanmean(

        np.stack(
            [
                u_wind_a,
                u_wind_d
            ],
            axis=0
        ),

        axis=0

    )


    v_wind = np.nanmean(

        np.stack(
            [
                v_wind_a,
                v_wind_d
            ],
            axis=0
        ),

        axis=0

    )


    variables = [

        sst,

        sss,

        ssh,

        u_current,

        v_current,

        u_wind,

        v_wind

    ]


    expected_shape = (
        PATCH_SIZE,
        PATCH_SIZE
    )


    for index, variable_data in enumerate(
        variables
    ):

        if variable_data.shape != expected_shape:

            raise RuntimeError(

                f"Local channel "
                f"{index} has shape "
                f"{variable_data.shape}. "

                f"Expected "
                f"{expected_shape}"

            )


    result = np.stack(

        variables,

        axis=0

    ).astype(
        np.float32
    )


    print(
        "Online patch shape:",
        result.shape
    )


    return result


# ============================================================
# INFORMATION
# ============================================================

if __name__ == "__main__":

    print(
        "OceanEmbed online data module"
    )

    print(
        f"Domain: "
        f"{LAT_MIN}-{LAT_MAX} N"
    )

    print(
        f"Domain: "
        f"{LON_MIN}-{LON_MAX} E"
    )

    print(
        "Grid:",
        len(TARGET_LAT),
        "x",
        len(TARGET_LON)
    )

    print(
        "Patch:",
        PATCH_SIZE,
        "x",
        PATCH_SIZE
    )