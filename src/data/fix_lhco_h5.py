import sys

import numpy as np
import tables


def fix(path: str) -> int:
    f = tables.open_file(path, "a")
    try:
        n = 0
        for node in f.walk_nodes("/"):
            attrs = node._v_attrs
            for name in list(attrs._f_list("all")):
                v = attrs[name]
                if isinstance(v, (bytes, np.bytes_)):
                    attrs[name] = bytes(v).decode("utf-8")
                    n += 1
    finally:
        f.close()
    return n


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) > 1
            else "develop/dataset/lhco/events_anomalydetection.h5")
    n = fix(path)
    print(f"decoded {n} byte-attrs to str in {path}")
