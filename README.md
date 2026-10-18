# Krait (Research Prototype)

> ⚠️ **Status: Active Research Prototype.**  
> An experimental, cross-platform genomics alignment engine exploring how much dynamic programming throughput can be extracted from consumer and legacy hardware (CPU multi-threading + portable WebGPU compute).

The goal of this project is to test whether low-end or older hardware (like 10-year-old laptops or integrated GPUs) can handle heavy, production-grade genomic workloads (BWA-MEM compatible affine-gap alignment) without vendor-locked ecosystems (CUDA) by leveraging **Rust** and **WebGPU/WGSL**.

---

## Benchmark (Real-World Data)

End-to-end alignment benchmark (from compressed `.fastq.gz` on disk through seeding, GPU compute, CIGAR generation, to formatted SAM output):

* **Hardware:** NVIDIA GeForce GTX 960M (Mobile Maxwell, 640 CUDA cores, 2015 laptop)
* **CPU:** Intel Core i7-5700HQ (4 cores / 8 threads @ 2.7 GHz)
* **Dataset:** *E. coli* K-12 (~4.64 Mbp) vs. 9.82M real Illumina reads (FASTQ, paired-capable)
* **End-to-End Wall Time:** `31.5 s`
* **Throughput:** **~311,000 reads/sec** (~1.25 GCUPS actual banded compute / equivalent to ~4.0 GCUPS full-matrix Gotoh)
* **Correctness:** Verified coordinates, CIGAR operations (M/I/D/S), and Phred-scaled MAPQ against standard SAM specifications.

---

## Key Architecture & Optimizations

The engine is built around a non-blocking asynchronous pipeline designed to keep both CPU cores and the GPU saturated simultaneously:

```
                 [ FASTQ.GZ Stream ]
                         │
                         ▼ 
           (Streaming zlib decompression)
┌────────────────────────────────────────────────────────┐
│ CPU Producer (Rayon Parallelized):                     │
│ • K-mer seed extraction & hash table lookup            │
│ • Candidate locus generation (up to 3 per read)        │
│ • Zero-allocation 2-bit & N-mask stack packing         │
│ • Transposition into Structure-of-Arrays (SoA)         │
└────────────────────────────────────────────────────────┘
                         │
                         ▼ 
    (Asynchronous double-buffering / PCIe ping-pong)
┌────────────────────────────────────────────────────────┐
│ GPU Execution (wgpu / WGSL):                           │
│ • Pass 1: Banded Gotoh DP Kernel (W = 32)              │
│   - Sheared coordinate transformation (k = j - i)      │
│   - In-place L1-resident 33-cell recurrence            │
│   - 4-bit packed direction & extension flags           │
│   - Z-drop heuristic early exit                        │
│ • Pass 2: Traceback & CIGAR Kernel                     │
│   - GPU-side winner pruning (losers exit in 3 cycles)  │
│   - Perfect-match short-circuit (skips VRAM reads)     │
│   - Finite-state-machine CIGAR reconstruction          │
└────────────────────────────────────────────────────────┘
                         │
                         ▼ 
              (Mapped staging buffers)
┌────────────────────────────────────────────────────────┐
│ CPU Writer Task (Rayon Worker Pool):                   │
│ • Suboptimal score extraction & MAPQ calculation       │
│ • Binary search contig coordinate resolution           │
│ • Streaming SAM/BAM generation                         │
└────────────────────────────────────────────────────────┘
```

### 1. Banded Gotoh with Sheared Coordinates ($W = 32$)
Instead of calculating the full quadratic $100 \times 128$ matrix (12,800 cells), the kernel straightens the diagonal band into a compact $100 \times 32$ rectangle using the transform $j = i + k$.
* Cuts cell calculations by **4x** (3,200 cells per read).
* Reduces DP array sizes to **33 elements** (264 bytes per thread), fitting entirely within the fast on-chip L1 cache and eliminating register spilling.
* Squeezes the global traceback directions buffer from 320 MB down to **80 MB** per batch.

### 2. Kernel Fission (Split DP & Traceback)
To maximize register occupancy, dynamic programming and backtracking are split into two specialized compute pipelines executed sequentially within a single command encoder:
* **`dp_kernel`**: Pure compute-bound matrix calculation with zero CIGAR buffer overhead.
* **`traceback_kernel`**: Lightweight FSM traceback with zero 1000-byte DP arrays.
* **GPU-Side Pruning**: Competing candidate loci of the same read evaluate scores in L1/L2 cache before traceback; losing candidates terminate immediately without reading VRAM directions.

### 3. Asynchronous Double-Buffering (Ping-Pong)
The GPU actor maintains two completely isolated hardware transport slots. While the GPU is computing batch $N$, the CPU is streaming batch $N+1$ across PCIe, and the writer is unmapping batch $N-1$. Hardware utilization stays balanced at ~65% CPU and ~90% GPU.

---

## Roadmap

- [x] Full-pipeline end-to-end streaming architecture (Tokio + Channels)
- [x] Gotoh affine gap scoring ($M, I, D$ state model)
- [x] Banded alignment ($W = 32$) with sheared coordinates
- [x] 4-bit state-aware traceback encoding
- [x] Multi-threaded zero-allocation CPU producer (Rayon)
- [x] Two-pass kernel fission with GPU-side candidate pruning
- [x] Multi-contig genome support with padded boundaries
- [x] Phred MAPQ mapping quality scoring
- [x] Valid SAM v1.6 output stream
- [ ] Paired-End (FR orientation) support & mate rescue
- [ ] Direct Uncompressed BAM (`uBAM`) streaming via `noodles`
- [ ] CPU SIMD acceleration for 2-bit packing (`fearless_simd`)
- [ ] Comprehensive error handling (replace `unwrap()` with typed errors)
- [ ] Golden-standard differential fuzzing against BWA-MEM / samtools

---

## Tech Stack

* **Language:** Rust (2024 edition)
* **GPU Compute:** `wgpu` (WGSL compute shaders, Vulkan / Metal / DirectX backends)
* **Concurrency:** `tokio` (async actor pipeline), `rayon` (data-parallel CPU workers)
* **Decompression:** `flate2` with `zlib-rs`
* **Profiling:** `perf`, `hotspot`

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

* Free for academic, research, and open-source use (including open-source network services).
* If you wish to use this software in proprietary/closed-source commercial products or network services (SaaS) without AGPLv3 copyleft obligations, custom commercial licenses are available upon request. 

Contact: **Valentyn Zhukovetskyi** - [zukovetski@gmail.com](mailto:zukovetski@gmail.com)
