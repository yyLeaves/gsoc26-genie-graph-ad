"""One-off repair for old Pandas HDF5 files with byte-valued attributes."""

import argparse

import numpy as np
import tables


def decode_byte_attrs(path: str) -> int:
    """Decode byte / numpy-bytes HDF5 attributes to UTF-8 strings in place."""
    handle = tables.open_file(path, "a")
    try:
        count = 0
        for node in handle.walk_nodes("/"):
            attrs = node._v_attrs
            for name in list(attrs._f_list("all")):
                value = attrs[name]
                if isinstance(value, (bytes, np.bytes_)):
                    attrs[name] = bytes(value).decode("utf-8")
                    count += 1
    finally:
        handle.close()
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode byte-valued attributes in an LHCO-style HDF5 file."
    )
    parser.add_argument("h5_path", help="Path to the HDF5 file to repair")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = decode_byte_attrs(args.h5_path)
    print(f"decoded {n} byte-attrs to str in {args.h5_path}")
