"""Build a multi-country YouTube trending dataset for one-step experiments.

State design:
- x_t is the per-country, per-day category share on the trending chart
- v_t is a per-country, per-day, per-category payoff proxy derived from engagement signals and centered across categories

This script aligns countries on a common daily axis and exports tensors:
    shares:          (C, T, N)
    payoff_*:        (C, T, N)
    payoff_obs_mask: (C, T, N)

The paper protocol uses the 15 categories common to all 10 countries over 122 aligned days.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import csv
import json
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
import fcntl
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

from pa_rmd.paths import PROJECT_ROOT
from pa_rmd.experiment_io import file_sha256


DEFAULT_COUNTRIES = ["US", "GB", "DE", "FR", "IN", "JP", "KR", "MX", "RU", "CA"]

KAGGLE_SOURCE = "datasnaek/youtube-new"
KAGGLE_VERSION = 115
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/datasnaek/youtube-new?datasetVersionNumber=115"
# SHA256 of the exact 20 raw files used by the paper, verified against v115.
RAW_SHA256 = {
    "CA_category_id.json": "b84afd7b659df4d7fa74ae7fbdfceb83fbacaafe8d857c7cc358fefed0b73205",
    "CAvideos.csv": "e7ff7f574d1e3d685e5a9406cde3ed2aa0b83d8f07d6d989e03c36d815f430b3",
    "DE_category_id.json": "b84afd7b659df4d7fa74ae7fbdfceb83fbacaafe8d857c7cc358fefed0b73205",
    "DEvideos.csv": "d0685b5281a295bdb5cf114a278696a681c7feac1172ad523c31461beb1980ea",
    "FR_category_id.json": "b84afd7b659df4d7fa74ae7fbdfceb83fbacaafe8d857c7cc358fefed0b73205",
    "FRvideos.csv": "cd318029a4b6d4ddd5f8ec644a790587e4fe692317e4f4adc7be503de7fbbb17",
    "GB_category_id.json": "80e23a959a55a27beed84be5f87fe0833baa8c29d53e64bbae23e5f6d4fb8fbc",
    "GBvideos.csv": "5f5c100e150be2baaf7f9a3198a8216ce107b7e03e4572790c15e698d8e763cb",
    "IN_category_id.json": "f857cd2b580ebe12e5b290f331fe2aac21ae4200ced4c9e03a317de47100752f",
    "INvideos.csv": "008313f3feb80dc94cbd3112437dfc9b4ab6a5c7bd5c892ab3c85e6582137b29",
    "JP_category_id.json": "f857cd2b580ebe12e5b290f331fe2aac21ae4200ced4c9e03a317de47100752f",
    "JPvideos.csv": "3dcc8a7a40f979c4f9a2431eb0f1061853a29086866d81de4fc8c714e1d38cf2",
    "KR_category_id.json": "f857cd2b580ebe12e5b290f331fe2aac21ae4200ced4c9e03a317de47100752f",
    "KRvideos.csv": "4567450dce9c3b489c7a949d1e10eaf70c999d58355f537374f8b0189580ce77",
    "MX_category_id.json": "f857cd2b580ebe12e5b290f331fe2aac21ae4200ced4c9e03a317de47100752f",
    "MXvideos.csv": "9cf987c1a9594a84839f2888474454c849f211e075fde9eb41f49c8620ac5a04",
    "RU_category_id.json": "f857cd2b580ebe12e5b290f331fe2aac21ae4200ced4c9e03a317de47100752f",
    "RUvideos.csv": "9ae2cf47b6f2ae3b3f5c35c6d34b052f2134df9f9ed744e930ed9acac725d556",
    "US_category_id.json": "2e892c5a5e48d284e40fd37de0313912264041aef8833e5323bd2b2fd08c7e25",
    "USvideos.csv": "09b4eb71295752705e472ebefeac9d2afab4177b7a818af795dea62744a48eb2"
}
PAPER_SHAPE = (10, 122, 15)


@dataclass
class CountryData:
    country: str
    category_counts_by_day: dict[str, Counter]
    metric_values_by_day: dict[str, dict[int, dict[str, list[float]]]]
    duplicate_rows_dropped: int
    unique_video_ids: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-country YouTube trending dataset.")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=DEFAULT_COUNTRIES,
        help="Country codes to include, e.g. US GB DE.",
    )
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Optional local raw directory; missing data is downloaded to a versioned cache here (default: output directory).")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data/YouTubeTrending/youtube_multicountry_common.npz"),
        help="Output .npz path.",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path):
    last_error = None
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                yield from csv.DictReader(f)
            return
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def parse_trending_date(raw: str) -> str:
    return datetime.strptime(raw, "%y.%d.%m").strftime("%Y-%m-%d")


def parse_nonnegative_int(raw: str) -> int:
    try:
        return max(int(raw), 0)
    except Exception:
        return 0


def load_category_name_map(countries: Iterable[str], data_dir: Path) -> dict[int, str]:
    category_name_map: dict[int, str] = {}
    for country in countries:
        category_path = data_dir / f"{country}_category_id.json"
        if not category_path.exists():
            continue
        with open(category_path, encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.get("items", []):
            cid = int(item["id"])
            category_name_map.setdefault(cid, item["snippet"]["title"])
    return category_name_map


def load_country_data(country: str, data_dir: Path) -> CountryData:
    csv_path = data_dir / f"{country}videos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing country CSV: {csv_path}")

    category_counts_by_day: dict[str, Counter] = defaultdict(Counter)
    metric_values_by_day: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_rows_dropped = 0
    unique_video_ids: set[str] = set()

    for row in read_csv_rows(csv_path):
        key = (row["video_id"], row["trending_date"])
        if key in seen_pairs:
            duplicate_rows_dropped += 1
            continue
        seen_pairs.add(key)
        unique_video_ids.add(row["video_id"])

        day = parse_trending_date(row["trending_date"])
        cid = int(row["category_id"])

        views = parse_nonnegative_int(row["views"])
        likes = parse_nonnegative_int(row["likes"])
        dislikes = parse_nonnegative_int(row["dislikes"])
        comments = parse_nonnegative_int(row["comment_count"])

        category_counts_by_day[day][cid] += 1

        stats = metric_values_by_day[day][cid]
        stats["views_raw"].append(np.log1p(views))
        stats["likes_raw"].append(np.log1p(likes))
        stats["dislikes_raw"].append(np.log1p(dislikes))
        stats["comments_raw"].append(np.log1p(comments))
        stats["like_rate"].append(np.log((likes + 1.0) / (views + 2.0)))
        stats["comment_rate"].append(np.log((comments + 1.0) / (views + 2.0)))
        stats["approval"].append(np.log((likes + 1.0) / (dislikes + 1.0)))
        denom = likes + dislikes
        if denom > 0:
            stats["like_dislike_balance"].append((likes - dislikes) / denom)
        else:
            stats["like_dislike_balance"].append(0.0)

    return CountryData(
        country=country,
        category_counts_by_day=dict(category_counts_by_day),
        metric_values_by_day={day: dict(by_cat) for day, by_cat in metric_values_by_day.items()},
        duplicate_rows_dropped=duplicate_rows_dropped,
        unique_video_ids=len(unique_video_ids),
    )


def select_category_ids(country_data: list[CountryData], mode: str) -> list[int]:
    actual_sets = []
    for item in country_data:
        ids = set()
        for counter in item.category_counts_by_day.values():
            ids.update(counter.keys())
        actual_sets.append(ids)
    if mode != "common":
        raise ValueError("The paper YouTube builder requires common categories.")
    selected = set.intersection(*actual_sets)
    return sorted(selected)


def compute_common_dates(country_data: list[CountryData]) -> list[str]:
    date_sets = [set(item.category_counts_by_day.keys()) for item in country_data]
    return sorted(set.intersection(*date_sets))


def fill_nan_forward_backward(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    t_len, n_dim = out.shape
    for j in range(n_dim):
        col = out[:, j]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size == 0:
            col[:] = 0.0
            continue
        first = valid[0]
        last = valid[-1]
        col[:first] = col[first]
        col[last + 1 :] = col[last]
        for a, b in zip(valid[:-1], valid[1:]):
            if b > a + 1:
                col[a + 1 : b] = col[a]
        out[:, j] = col
    return out


def build_arrays(
    country_data: list[CountryData],
    dates: list[str],
    category_ids: list[int],
) -> dict[str, np.ndarray]:
    num_countries = len(country_data)
    num_days = len(dates)
    num_categories = len(category_ids)

    shares = np.zeros((num_countries, num_days, num_categories), dtype=np.float32)
    payoff_obs_mask = np.zeros((num_countries, num_days, num_categories), dtype=bool)

    metric_names = [
        "views_raw",
        "likes_raw",
        "dislikes_raw",
        "comments_raw",
        "like_rate",
        "comment_rate",
        "approval",
        "like_dislike_balance",
    ]
    output_names = {
        "views_raw": "payoff_log_views",
        "likes_raw": "payoff_log_likes",
        "dislikes_raw": "payoff_log_dislikes",
        "comments_raw": "payoff_log_comments",
        "like_rate": "payoff_like_rate",
        "comment_rate": "payoff_comment_rate",
        "approval": "payoff_approval",
        "like_dislike_balance": "payoff_like_dislike_balance",
    }
    raw_metrics = {
        name: np.full((num_countries, num_days, num_categories), np.nan, dtype=np.float32)
        for name in metric_names
    }

    cid_to_idx = {cid: idx for idx, cid in enumerate(category_ids)}

    for c_idx, item in enumerate(country_data):
        for d_idx, day in enumerate(dates):
            counts = item.category_counts_by_day.get(day, Counter())
            total = sum(counts.get(cid, 0) for cid in category_ids)
            if total > 0:
                for cid in category_ids:
                    shares[c_idx, d_idx, cid_to_idx[cid]] = counts.get(cid, 0) / total

            day_metrics = item.metric_values_by_day.get(day, {})
            for cid, per_metric in day_metrics.items():
                if cid not in cid_to_idx:
                    continue
                idx = cid_to_idx[cid]
                payoff_obs_mask[c_idx, d_idx, idx] = True
                for metric_name in metric_names:
                    values = per_metric.get(metric_name, [])
                    if values:
                        raw_metrics[metric_name][c_idx, d_idx, idx] = float(np.median(values))

    outputs: dict[str, np.ndarray] = {
        "shares": shares,
        "payoff_obs_mask": payoff_obs_mask,
    }

    for metric_name, arr in raw_metrics.items():
        centered = np.zeros_like(arr)
        for c_idx in range(num_countries):
            filled = fill_nan_forward_backward(arr[c_idx])
            centered[c_idx] = filled - filled.mean(axis=1, keepdims=True)
        outputs[output_names[metric_name]] = centered.astype(np.float32)

    return outputs


def _array_sha256(value: np.ndarray) -> str:
    value = np.asarray(value)
    content = (json.dumps(value.tolist(), ensure_ascii=True, separators=(",", ":")).encode()
               if value.dtype.hasobject else value.tobytes(order="C"))
    header = json.dumps([value.dtype.str, list(value.shape)]).encode()
    return hashlib.sha256(header + b"\0" + content).hexdigest()


def _validate_npz(path: Path) -> None:
    """Accept legacy external NPZs; fully verify provenance on newly prepared ones."""
    try:
        if not zipfile.is_zipfile(path):
            raise ValueError("not a complete NPZ archive")
        with np.load(path, allow_pickle=True) as data:
            countries = [str(value) for value in data["countries"]]
            shares = np.asarray(data["shares"])
            if len(countries) != 10 or len(set(countries)) != 10 or shares.shape != PAPER_SHAPE:
                raise ValueError("requires 10 unique countries, 122 days and 15 categories")
            if (not np.isfinite(shares).all() or np.any(shares < 0)
                    or not np.allclose(shares.sum(axis=-1), 1.0, atol=1e-5)):
                raise ValueError("shares must be finite probability distributions")
            if "payoff_log_views" not in data.files:
                raise ValueError("missing payoff_log_views")
            for key in data.files:
                if key.startswith("payoff_"):
                    value = data[key]
                    if value.shape != PAPER_SHAPE or not np.isfinite(value).all():
                        raise ValueError(f"invalid shape or non-finite {key}")
            if "preparation_json" in data.files:
                record = json.loads(str(data["preparation_json"].item()))
                if not isinstance(record, dict):
                    raise ValueError("invalid preparation record")
                expected = {"schema": 1, "source": KAGGLE_SOURCE, "version": KAGGLE_VERSION,
                            "download_url": KAGGLE_URL, "license": "CC0-1.0",
                            "countries": DEFAULT_COUNTRIES,
                            "category_mode": "common", "shape": list(PAPER_SHAPE),
                            "raw_sha256": RAW_SHA256}
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError("preparation source or configuration mismatch")
                keys = set(data.files) - {"preparation_json"}
                if set(record.get("array_sha256", {})) != keys:
                    raise ValueError("preparation array inventory mismatch")
                if any(_array_sha256(data[key]) != record["array_sha256"][key] for key in keys):
                    raise ValueError("preparation array checksum mismatch")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"Invalid YouTube NPZ {path}: {exc}") from exc


@contextmanager
def _file_lock(path: Path):
    # flock is released automatically if a process exits; the harmless file remains.
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _check_raw(root: Path) -> bool:
    """Check every supplied paper file, including partial local raw directories."""
    complete = True
    for name, expected in RAW_SHA256.items():
        path = root / name
        if not path.exists():
            complete = False
        elif not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"YouTube raw checksum mismatch: {path}; expected Kaggle v115 paper files")
    return complete


def _download_archive(destination: Path) -> None:
    request = urllib.request.Request(KAGGLE_URL, headers={"User-Agent": "PA-RmD-data-preparation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > 600 * 1024 * 1024:
                    raise ValueError("Kaggle download exceeds the fixed dataset size limit")
                output.write(chunk)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not download public Kaggle YouTube v115 from {KAGGLE_URL}: {exc}") from exc


def _extract_raw(archive: Path, destination: Path) -> None:
    # Copy only named paper files; never extract archive paths supplied by the server.
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        for name in RAW_SHA256:
            matches = [item for item in names if Path(item).name == name]
            if len(matches) != 1:
                raise ValueError(f"Kaggle v115 archive requires exactly one {name}")
            member = source.getinfo(matches[0])
            if member.file_size > 100 * 1024 * 1024:
                raise ValueError(f"Unexpected Kaggle raw file size: {name}")
            with source.open(member) as incoming, (destination / name).open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    if not _check_raw(destination):
        raise ValueError("Incomplete Kaggle v115 raw archive")


def _ensure_raw(root: Path) -> Path:
    if _check_raw(root):
        return root
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".youtube-v115"
    with _file_lock(root / ".youtube-v115.lock"):
        if cache.exists():
            if not _check_raw(cache):
                raise ValueError(f"Incomplete YouTube cache: {cache}; move it aside before retrying")
            return cache
        print(f"Downloading public Kaggle {KAGGLE_SOURCE} version {KAGGLE_VERSION} ...")
        with tempfile.TemporaryDirectory(prefix=".youtube-download-", dir=root) as temporary:
            work = Path(temporary)
            archive = work / "dataset.zip"
            _download_archive(archive)
            raw = work / "raw"
            raw.mkdir()
            _extract_raw(archive, raw)
            # A nonempty existing cache is never overwritten.
            raw.rename(cache)
    return cache


def ensure_youtube_dataset(npz_path: str | Path, raw_root: str | Path | None = None) -> Path:
    """Reuse a valid NPZ, or atomically prepare it from verified local/public raw data.

    Existing external NPZs retain the loader's shape/finite/probability contract.
    NPZs built here also embed fixed source/version/raw and array checksums.
    Raw inputs are never overwritten. A failed preparation leaves no output NPZ.
    """
    output = Path(npz_path).expanduser().resolve()
    if output.exists():
        _validate_npz(output)
        return output
    if output.suffix != ".npz":
        raise ValueError("YouTube output must have a .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(output.with_name(output.name + ".lock")):
        if output.exists():
            _validate_npz(output)
            return output
        raw = _ensure_raw(Path(raw_root).expanduser().resolve() if raw_root is not None else output.parent)
        with tempfile.TemporaryDirectory(prefix=".youtube-build-", dir=output.parent) as temporary:
            staged = Path(temporary) / "dataset.npz"
            _build_npz(raw, staged)
            _validate_npz(staged)
            # link provides atomic publication and refuses to replace any user file.
            os.link(staged, output)
    return output


def _build_npz(data_dir: Path, output_path: Path) -> None:
    countries = list(DEFAULT_COUNTRIES)

    print(f"Building multi-country dataset for: {countries}")
    category_name_map = load_category_name_map(countries, data_dir)
    country_data = [load_country_data(country, data_dir) for country in countries]

    category_ids = select_category_ids(country_data, mode="common")
    dates = compute_common_dates(country_data)
    if len(category_ids) != 15 or len(dates) != 122:
        raise ValueError(
            f"The paper YouTube protocol requires 15 common categories and 122 aligned days; "
            f"found {len(category_ids)} categories and {len(dates)} days."
        )

    category_names = np.array(
        [category_name_map.get(cid, f"Category {cid}") for cid in category_ids],
        dtype=object,
    )
    arrays = build_arrays(country_data, dates, category_ids)

    metadata = {
        "countries": countries,
        "category_mode": "common",
        "num_countries": len(countries),
        "num_days": len(dates),
        "num_categories": len(category_ids),
        "duplicates_dropped": {
            item.country: item.duplicate_rows_dropped for item in country_data
        },
        "unique_video_ids": {
            item.country: item.unique_video_ids for item in country_data
        },
        "payoff_fields": [
            "payoff_log_views",
            "payoff_log_likes",
            "payoff_log_dislikes",
            "payoff_log_comments",
            "payoff_like_rate",
            "payoff_comment_rate",
            "payoff_approval",
            "payoff_like_dislike_balance",
        ],
    }

    payload = dict(
        countries=np.array(countries, dtype=object),
        dates=np.array(dates, dtype=object),
        category_ids=np.array(category_ids, dtype=np.int32),
        category_names=category_names,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True), dtype=object),
        **arrays,
    )
    preparation = {
        "schema": 1, "source": KAGGLE_SOURCE, "version": KAGGLE_VERSION,
        "download_url": KAGGLE_URL, "license": "CC0-1.0",
        "countries": countries, "category_mode": "common", "shape": list(PAPER_SHAPE),
        "raw_sha256": RAW_SHA256,
        "array_sha256": {key: _array_sha256(value) for key, value in payload.items()},
    }
    with output_path.open("wb") as stream:
        np.savez(stream, **payload, preparation_json=np.array(json.dumps(preparation, sort_keys=True)))

    print(f"Saved: {output_path}")
    print(f"  countries:   {len(countries)}")
    print(f"  days:        {len(dates)}")
    print(f"  categories:  {len(category_ids)}")
    print(f"  category ids: {category_ids}")
    print(f"  category names: {list(category_names)}")
    for item in country_data:
        print(
            f"  {item.country}: unique_videos={item.unique_video_ids} "
            f"duplicates_dropped={item.duplicate_rows_dropped}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    countries = [country.strip().upper() for country in args.countries if country.strip()]
    if countries != DEFAULT_COUNTRIES:
        raise ValueError("The paper YouTube builder requires the default ten-country order.")
    print(f"Ready: {ensure_youtube_dataset(args.output, args.raw_root)}")


if __name__ == "__main__":
    main()
