# Data

The C-MAPSS records are not redistributed here. They are public and are downloaded from the NASA Prognostics Center of Excellence data repository (Turbofan Engine Degradation Simulation, C-MAPSS), which is the source cited in the manuscript.

Place the files under `exp/data_cmapss/` with the names below. The checksums are what the reported numbers were produced from.

| file | bytes | sha256 |
|---|---|---|
| `train_FD001.txt` | 3515356 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | 2228855 | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | 429 | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `train_FD002.txt` | 9082480 | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `test_FD002.txt` | 5734587 | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `RUL_FD002.txt` | 1110 | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| `train_FD004.txt` | 10350705 | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |

`train_FD004.txt` is used only for the structural checks in `diagnostics/diag_multicond.py`; no result in the paper depends on FD004, and neither `test_FD004.txt` nor `RUL_FD004.txt` is needed.
