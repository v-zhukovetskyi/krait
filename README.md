# Krait (banded-krait)

[![Crates.io](https://img.shields.io/crates/v/banded-krait.svg)](https://crates.io/crates/banded-krait)
[![License: AGPL-3.0](https://img.shields.io/crates/l/banded-krait.svg)](https://www.gnu.org/licenses/agpl-3.0)

Project Status (v0.1.0):
Functional Alpha Prototype. The core WebGPU alignment engine is stable and validated on bacterial datasets.

The purpose of this project is to evaluate whether older systems (such as legacy laptops or integrated graphics) can process high-throughput genomic workloads (BWA-MEM compatible affine-gap alignment) without vendor-locked GPU frameworks (CUDA) by using Rust and WebGPU/WGSL.

> **Current Prototype Scope:**  
> The compute shaders in this release are specialized for **fixed-length 100 bp reads** (standard Illumina 2x100 PE libraries). Read lengths differing from 100 bp are currently rejected at the ingestion stage. Support for variable read lengths (dynamic binning up to 250 bp) is targeted for the next release (see Roadmap).

---
## Benchmarks

### Test Setup
* Hardware: Intel Core i7-5700HQ (4 cores / 8 threads @ 2.69 GHz under thermal constraints), NVIDIA GeForce GTX 960M (2 GB VRAM, Maxwell GM107, 640 CUDA cores)
* Operating System: Fedora Linux 44 x86_64
* Reference: Escherichia coli K-12 str. MG1655 (NC_000913.3, 4.64 Mbp)
* Input Dataset: 44,928,012 paired-end Illumina reads (ERR022075, ~12 GB uncompressed)
* Execution: Complete pipeline from compressed .fastq.gz inputs directly to disk-bound SAM format. Measured via GNU time (/usr/bin/time -v) and nvidia-smi.

### Performance Comparison

| Tool | Implementation | Wall Time | User Time | CPU % | Peak RAM | GPU % (Avg/Peak) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Krait | WebGPU + 8 CPU threads | 1m 03.8s | 181.5 s | 303% | 1.37 GB | 72.8% / 100% | 716 MB |
| minibwa | CPU (8 threads, AVX2) | 2m 43.7s | 846.3 s | 568% | 1.46 GB | N/A | N/A |
| minimap2 (2.31) | CPU (8 threads, -ax sr) | 3m 14.8s | 1064.7 s | 562% | 0.69 GB | N/A | N/A |
| bwa-mem2 (2.2.1) | CPU (8 threads, AVX2) | 4m 58.9s | 1918.1 s | 654% | 1.05 GB | N/A | N/A |
| bwa-mem (0.7.19) | CPU (8 threads, scalar) | 5m 47.6s | 2338.6 s | 681% | 0.99 GB | N/A | N/A |

### Concordance vs. BWA-MEM

Evaluated across all 44,928,012 read pairs using read-name hash pairing against standard BWA-MEM output:

| Metric | Records | Percentage |
| :--- | :--- | :--- |
| Total Evaluated Reads | 44,928,012 | 100.00% |
| Concordant Placements (<= 5 bp) | 43,550,574 | 99.26% |
| Exact Coordinate Match (0 bp difference) | 43,437,482 | 99.00% |
| Near Match (<= 5 bp difference, gap shift) | 113,092 | 0.26% |
| Discrepant Coordinate (> 5 bp difference, repetitive loci) | 324,562 | 0.74% |
| False-Positive Mappings (mapped in Krait, unmapped in BWA) | 22 | 0.00% |

SAM validation (samtools flagstat):
* Total passed: 44,928,012
* Mapped: 43,867,824 (97.64%)
* Properly paired: 43,118,740 (95.97%)
* Singletons: 749,084 (1.67%)
* Secondary / Supplementary: 0 / 0

---
## Installation

### 1. Precompiled Standalone Binaries (No Rust or CUDA Required)
Download the latest static binary for your platform from the [**Releases Page (v0.1.0)**](https://github.com/v-zhukovetskyi/krait/releases/tag/v0.1.0):

* **Linux (x86_64, static musl):** [`krait-linux-x86_64`](https://github.com/v-zhukovetskyi/krait/releases/download/v0.1.0/krait-linux-x86_64)
* **macOS (Apple Silicon M1/M2/M3/M4):** [`krait-macos-arm64`](https://github.com/v-zhukovetskyi/krait/releases/download/v0.1.0/krait-macos-arm64)
* **Windows (x86_64):** [`krait-windows-x86_64.exe`](https://github.com/v-zhukovetskyi/krait/releases/download/v0.1.0/krait-windows-x86_64.exe)

Quick install for Linux/macOS via terminal:
```bash
# Linux
curl -L -o krait https://github.com/v-zhukovetskyi/krait/releases/download/v0.1.0/krait-linux-x86_64
chmod +x krait && sudo mv krait /usr/local/bin/

# macOS (Apple Silicon)
curl -L -o krait https://github.com/v-zhukovetskyi/krait/releases/download/v0.1.0/krait-macos-arm64
chmod +x krait && sudo mv krait /usr/local/bin/
```

### 2. From Crates.io
If you have the Rust toolchain installed:
```bash
cargo install banded-krait
```

### 3. Build from Source
```bash
git clone https://github.com/v-zhukovetskyi/krait.git
cd krait
cargo build --release
```

### System Requirements
* **GPU Drivers:** Any modern graphics driver supporting **Vulkan** (Linux/Windows), **Metal** (macOS), or **DirectX 12**.
* **Zero External Dependencies:** No CUDA Toolkit, no dynamic library runtime hell, and no Python/Conda environment required.

---

## Pipeline Architecture

Krait executes as an in-order pipeline of dedicated OS threads communicating over bounded crossbeam channels:

```
[ FASTQ.GZ Inputs ]
   │
   ├──────────────────────────────┐
   ▼                              ▼
Reader Thread R1               Reader Thread R2
(Decompress + SIMD memchr)     (Decompress + SIMD memchr)
   │                              │
   └──────────────┬───────────────┘
                  ▼
          Producer Thread
   (Rayon Worker Pool, coarse-grained chunks)
   • Seed extraction & hash table lookup
   • FR orientation resolution & mate rescue
   • Contiguous task arena serialization
                  │
                  ▼
          Aggregator Thread
   • Column-major Structure-of-Arrays (SoA) layout
   • Candidate family preservation across batches
                  │
                  ▼
          GPU Worker Thread
   (wgpu double-buffered ping-pong execution slots)
   • Pass 1: Banded Gotoh DP Kernel (W = 32, sheared coords)
   • Pass 2: Traceback & CIGAR Kernel (winner pruning)
   • Host-visible staging readback
                  │
                  ▼
          Writer Thread
   • Zero-allocation chunked byte formatting
   • Automated trailing soft-clip harmonization
   • Buffered SAM v1.6 output
```


## Benchmarks

### Core Design Decisions

1. Synchronous OS Threading Model  
   All stages run on dedicated OS threads (`std::thread`) connected via `crossbeam::channel::bounded`. Async runtimes are avoided in the data path to prevent executor overhead, stack-frame state-machine allocations, and CPU cache eviction across worker threads. GPU runtime initialization uses `pollster::block_on`.

2. Concurrent Two-Stream Decompression  
   Paired-end inputs (R1 and R2) run in separate threads, eliminating single-core decompression bottlenecks on standard GZIP inputs. Raw byte streams are parsed in bulk blocks via SIMD `memchr`, avoiding per-line function calls.

3. Banded Gotoh with Sheared Coordinates ($W = 32$, Fixed 100 bp Tile)  
   Straightens diagonal alignment sweeps into compact $100 \times 32$ matrices ($k = j - i$). Hardcoding the read tile height to 100 bp eliminates warp-divergent boundary branches in the hot loop, maximizes register allocation per thread, and keeps DP recurrence state strictly within 33 registers/L1 cache.

4. Kernel Fission with Workgroup Pruning  
   Separates dynamic programming (`dp_kernel`) and traceback (`traceback_kernel`) into sequential compute passes within a single command encoder. Read candidate sets evaluate scores in registers; sub-optimal candidates exit immediately without reading direction matrices from VRAM.

5. Dynamic WGSL Template Injection  
   Pipeline constants (workgroup size, band size, word dimensions, scoring thresholds) are defined once in Rust and injected into shader source files at runtime via template markers, ensuring compile-time synchronization without manual synchronization.

6. Zero-Allocation SAM Serialization  
   Output generation operates on coarse-grained chunks (512 records per Rayon worker) serializing directly into raw byte buffers. Avoids intermediate heap `String` and `Vec` allocations during CIGAR generation, sequence reversal, and flag construction.

---

## CLI Options

```
Usage: krait [OPTIONS] -r <FASTA> -1 <FASTQ>

Input / Output:
  -r, --reference <FASTA>   Reference genome file (FASTA, optional GZ)
  -1, -i, --reads <FASTQ>   First-mate FASTQ file (currently requires fixed 100 bp reads)
  -2, --reads2 <FASTQ>      Second-mate FASTQ file (currently requires fixed 100 bp reads)
  -o, --output <FILE>       Output alignment file path ('-' for stdout) [default: -]

Algorithm & Indexing:
  -k <INT>                  K-mer length for seed index [limit: 16, default: 15]

Execution & Performance:
  -b <INT>                  Target batch size for GPU slots [default: 100000]
  -t, --threads <INT>       Worker threads for parallel tasks (0 = auto) [default: 0]
  -l <INT>                  Read processing limit (0 = process all) [default: 0]
  -p, --progress            Display progress and throughput metrics
  -h, --help                Print help information
```

---

## Roadmap

- [x] Dedicated OS thread pipeline with bounded crossbeam channels
- [x] Multi-file concurrent decompression (R1/R2)
- [x] Bulk FASTQ block decompression with SIMD memchr parsing
- [x] Dual-pass WGSL compute pipeline (Banded DP + Traceback)
- [x] Double-buffered execution slots (ping-pong)
- [x] Illumina FR paired-end mapping with mate rescue
- [x] Zero-allocation chunked SAM serialization
- [x] Shared constant injection into WGSL templates
- [ ] Replacement of k-mer hash table with minimizer index (minimap2-style)
- [ ] Scaling index memory footprint for human reference genomes (GRCh38)
- [ ] Dynamic MAPQ calculation based on secondary alignment score margins
- [ ] Direct uncompressed BAM (uBAM) output streaming via noodles
- [ ] Variable read length support across arbitrary sequencing technologies

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3).

* Free for academic, educational, and open-source use under AGPLv3 copyleft terms.
* Commercial licensing without AGPLv3 copyleft requirements is available upon request.

Contact: Valentyn Zhukovetskyi - zukovetski@gmail.com
