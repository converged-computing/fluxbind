### Hardware Profile: Google Cloud c4-standard-96

| Feature | Value | Explanation |
| :--- | :--- | :--- |
| **CPU Architecture** | Intel Sapphire Rapids | A tiled architecture presented as a single socket. |
| **Physical Sockets** | 1 (`Package L#0`) | The entire CPU is a single physical package. |
| **NUMA Nodes** | **2** | The CPU is divided into two distinct NUMA domains (`NUMANode L#0`, `L#1`). |
| **Total Physical Cores** | **48** | The total number of high-performance physical cores. |
| **Cores per NUMA Node** | **24** | Cores `L#0-L#23` are on `NUMANode L#0`; Cores `L#24-L#47` are on `NUMANode L#1`. |
| **SMT / Hyper-Threading** | Enabled | Each physical core presents 2 Processing Units (PUs) to the OS. |
| **Total PUs (vCPUs)** | **96** | The total number of schedulable hardware threads (`PU L#0` to `L#95`). |
| **PU Numbering Scheme** | Split / Interleaved | The first 48 PUs (`P#0-P#47`) are the first threads on each of the 48 cores. The second 48 PUs (`P#48-P#95`) are the second threads. |
| **L3 Cache** | 260 MB | A single, large L3 cache is shared across all cores. |
| **L2 Cache** | 2048 KB (2 MB) | Each physical core has its own private 2 MB L2 cache. |
| **L1d / L1i Cache** | 48 KB / 32 KB | Each physical core has its own private L1 data and instruction caches. |
