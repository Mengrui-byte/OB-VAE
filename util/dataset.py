"""Order-book windows from one uncompressed NPZ per asset."""

import json
import struct
import zipfile
from datetime import date
from operator import index as integer_index
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class OrderBookDataset(Dataset):
    def __init__(
        self, data=None, window_size=256, stride=1, *, asset=None,
        start=None, end=None, npz_dir="/mnt/Data/mengrui/2048_snapshot_dataset",
        preload=False, return_counts=False,
    ):
        """返回 float32 [window_size,2,2048,2]，特征为实际价格、数量。

        每个币种对应一个文件，例如 npz_dir/BTCUSDT.npz，包含完整 time/data。
        npz_dir 也可以直接传入该 NPZ 文件路径。
        start/end 为 UTC 日期或整数微秒，范围 [start,end)，窗口不跨缺失分钟。
        默认内存映射，只读当前窗口；preload=True 时将选定区间复制进内存。
        价量缩放由训练循环完成，NPZ 保存实际单位。
        return_counts=True 时返回 data 和 level_counts，供损失屏蔽补零档位。
        data 参数兼容直接传入 [N,2,2048,F] Tensor 的方式。
        """
        if isinstance(window_size, bool) or isinstance(stride, bool):
            raise ValueError("window_size and stride must be positive integers")
        self.window_size = integer_index(window_size)
        self.stride = integer_index(stride)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive integers")
        self.data = data
        self.preload = preload
        self.return_counts = return_counts
        if data is not None:
            if any(value is not None for value in (asset, start, end)):
                raise ValueError("data cannot be combined with NPZ date/asset parameters")
            return
        if asset is None or start is None or end is None:
            raise ValueError("asset, start and end are required when data is omitted")
        symbol = asset.split("/")[-1].upper()
        if symbol not in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"):
            raise ValueError(f"Unsupported asset: {asset}")
        self.asset = f"binance-futures/{symbol}"
        root = Path(npz_dir).expanduser().resolve()
        path = root if root.suffix == ".npz" else root / f"{symbol}.npz"
        self.index_path = str(path)
        if not path.is_file():
            raise FileNotFoundError(f"No complete NPZ: {path}; run runs/export_snapshots.sh first")
        bounds = []
        for value in (start, end):
            if isinstance(value, str):
                day = date.fromisoformat(value)
                if day.isoformat() != value:
                    raise ValueError("date must be YYYY-MM-DD")
                value = (day - date(1970, 1, 1)).days * 86_400_000_000
            if isinstance(value, bool):
                raise ValueError("time must be a UTC date or integer microseconds")
            value = integer_index(value)
            if value < 0:
                raise ValueError("time must be nonnegative")
            bounds.append(value)
        first, stop = bounds
        if stop < first:
            raise ValueError("start must not exceed end")
        with np.load(path, allow_pickle=False) as stored:
            if stored["asset"].item() != self.asset or stored["schema_version"].item() != 1 or stored["bar_seconds"].item() != 60:
                raise ValueError(f"Invalid NPZ schema/asset: {path}")
            times = stored["time"]
            metadata = json.loads(stored["metadata"].item())
            counts = stored["level_counts"] if return_counts else None
        if times.dtype != np.int64 or times.ndim != 1 or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid NPZ timestamps: {path}")
        if metadata.get("status") != "complete":
            raise ValueError(f"Incomplete NPZ archive: {path}")
        if first < int(np.datetime64(metadata["requested_start"], "us").astype(np.int64)) or stop > int(np.datetime64(metadata["available_end"], "us").astype(np.int64)):
            raise ValueError(f"Requested dates exceed NPZ coverage: {path}")
        # np.load(..., mmap_mode=...) does not memory-map ZIP members.
        # Locate the standard, uncompressed data.npy member and map its payload.
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("data.npy")
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
                raise ValueError("Dataset requires an uncompressed, unencrypted NPZ data member")
        with path.open("rb") as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid ZIP local header: {path}")
            fields = struct.unpack("<4s5H3I2H", header)
            member_offset = info.header_offset + 30 + fields[-2] + fields[-1]
            stream.seek(member_offset)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY format: {version}")
            self._data_offset = stream.tell()
        if shape != (len(times), 2, 2048, 2) or dtype != np.float32 or fortran:
            raise ValueError(f"Invalid NPZ data shape, order or dtype: {path}")
        expected_bytes = self._data_offset - member_offset + len(times) * 2 * 2048 * 2 * 4
        if expected_bytes != info.file_size or self._data_offset + len(times) * 32768 > path.stat().st_size:
            raise ValueError(f"Truncated NPZ data: {path}")
        self._shape = shape
        begin, finish = np.searchsorted(times, [first, stop])
        self._source_start = int(begin)
        self.times = times[begin:finish]
        if return_counts:
            if counts.shape != (len(times), 2) or counts.dtype != np.uint16 or np.any(counts > 2048):
                raise ValueError(f"Invalid NPZ level counts: {path}")
            self.level_counts = counts[begin:finish]
        candidates = np.arange(0, max(0, len(self.times) - self.window_size + 1), self.stride, dtype=np.int64)
        gaps = np.concatenate(([0], np.cumsum(np.diff(self.times // 60_000_000) != 1)))
        self.starts = candidates[gaps[candidates + self.window_size - 1] == gaps[candidates]]

    def __len__(self):
        """能够截取出的完整、连续分钟窗口数量。"""
        if hasattr(self, "starts"):
            return len(self.starts)
        return max(0, (len(self.data) - self.window_size) // self.stride + 1)

    def __getitem__(self, index):
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("OrderBookDataset index out of range")
        start = int(self.starts[index]) if hasattr(self, "starts") else index * self.stride
        end = start + self.window_size
        if self.data is not None:
            result = self.data[start:end]
            if not self.return_counts:
                return result
            counts = (torch.from_numpy(self.level_counts[start:end].astype(np.int64))
                      if hasattr(self, "level_counts") else (result[..., 0] > 0).sum(dim=-1))
            return {"data": result, "level_counts": counts}
        # Keep no open file/mmap handle on the Dataset, so workers can pickle it.
        mapped = np.memmap(self.index_path, dtype=np.float32, mode="r",
                           offset=self._data_offset, shape=self._shape)
        if self.preload:
            result = torch.empty((len(self.times), 2, 2048, 2), dtype=torch.float32)
            target = result.numpy()
            with tqdm(total=len(self.times), desc="读取 NPZ", unit="分钟", dynamic_ncols=True) as progress:
                for offset in range(0, len(self.times), 1440):
                    stop = min(offset + 1440, len(self.times))
                    np.copyto(target[offset:stop], mapped[self._source_start + offset:self._source_start + stop])
                    if not np.isfinite(target[offset:stop]).all():
                        raise ValueError(f"Nonfinite NPZ data: {self.index_path}")
                    progress.update(stop - offset)
            self.data = result
            result = self.data[start:end]
        else:
            result = torch.from_numpy(np.array(mapped[self._source_start + start:self._source_start + end], copy=True))
        if not np.isfinite(result.numpy()).all():
            raise ValueError(f"Nonfinite NPZ window: {self.index_path}")
        if self.return_counts:
            return {"data": result, "level_counts": torch.from_numpy(self.level_counts[start:end].astype(np.int64))}
        return result


if __name__ == "__main__":
    dataset = OrderBookDataset(data=torch.randn(260, 2, 2048, 2))
    print(dataset[0].shape)
    print(next(iter(DataLoader(dataset, batch_size=2))).shape)
