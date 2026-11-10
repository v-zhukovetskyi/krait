## Technical Specification (v0.1)

This specification defines the algorithmic models, memory layouts, binary representations, and pipeline protocols implemented in Krait v0.1.

---

### 1. Sequence Representation & Ambiguity Handling

* **2-bit Nucleotide Alphabet:**
  Canonical nucleotides (`A`, `C`, `G`, `T` and lowercase equivalents `a`, `c`, `g`, `t`) are mapped to 2-bit unsigned integers:
  $$\text{A} = 00_2 \ (0), \quad \text{C} = 01_2 \ (1), \quad \text{G} = 10_2 \ (2), \quad \text{T} = 11_2 \ (3)$$
* **Storage Packing:**
  Nucleotides are packed into 32-bit unsigned words (`u32`) in little-endian bit order:
  $$\text{word} = \sum_{i=0}^{15} \text{base}_i \ll (2 \times i)$$
  - Reads ($L \le 100\text{ bp}$): Packed into 7 contiguous `u32` words ($16 \times 7 = 112\text{ bases capacity}$).
  - Reference windows ($L = 128\text{ bp}$): Packed into 8 contiguous `u32` words ($16 \times 8 = 128\text{ bases capacity}$).
* **Ambiguity & IUPAC Masking:**
  All non-canonical characters (ambiguous base `N`, IUPAC degenerate codes `R, Y, S, W, K, M, B, D, H, V`, gaps, and soft-clipped bases) are represented in a parallel 1-bit mask array:
  - Mask bit = $1$: Base is ambiguous/unknown.
  - Mask bit = $0$: Base is canonical.
  - Packed into `u32` words at 32 bases per word (4 words per 100 bp read).
* **Scoring Evaluation:**
  Any cell comparison where either the query or target mask bit is $1$ evaluates strictly to `mismatch_penalty` ($-3$), regardless of the underlying 2-bit value.

---

### 2. Reference Genome Layout & Chromosomal Boundaries

* **Linear Concatenation:**
  Multi-chromosome FASTA inputs are flattened into a single contiguous byte array.
* **Padding Buffer:**
  A fixed padding buffer of $256$ ambiguous bases (`N`, `PADDING_LEN`) is appended:
  1. Immediately following each chromosome sequence.
  2. At the trailing end of the final chromosome.
* **Out-of-Bounds Invariant:**
  The trailing $256\text{ bp}$ padding guarantees that reference extraction slices of length $L = 128\text{ bp}$ (`REF_LEN`) never exceed the physical allocation bounds of the genome buffer, even if a seed hit occurs on the terminal base of the last contig.
* **Coordinate Resolution:**
  Global linear coordinates ($0$-based) are mapped to contig names and $1$-based local positions via binary search over contig boundary offsets (`global_start`). Hits falling within the $256\text{ bp}$ inter-chromosomal padding intervals are designated unresolvable and marked unmapped.

---

### 3. Seeding & Pair Candidate Discovery

* **K-mer Parameters:**
  Seed length $k = 15$ ($30\text{ bits}$ binary representation, fitting within a single `u32`).
* **K-mer Masking Filter:**
  Any $15$-mer containing soft-masked lowercase bases or 'N' characters is discarded during both indexing and query lookups.
* **Index Data Structure:**
  Non-repetitive k-mers are indexed into an in-memory hash table (`FxHashMap<u32, u32>`). K-mers occurring more than once are invalidated during indexing (`u32::MAX`) to restrict seeding strictly to unique anchor loci.
* **Tripartite Query Sampling:**
  For each $100\text{ bp}$ query read, candidate anchors are evaluated at three fixed offsets:
  $$\text{Offsets} = [5, \; 42, \; 80]$$
  Offset $5$ provides deliberate clearance against $5'$-adapter noise and terminal $N$-calls.
* **Seed Projection:**
  Seed hits are projected to the beginning of the local reference alignment window using a symmetric flanking offset $\text{FLANK} = 14\text{ bp}$:
  $$\text{ref\_start} = (\text{hit\_pos} - \text{seed\_offset} - \text{FLANK})$$
* **Pair Geometry (FR Orientation):**
  Pairs are matched using an $O(N+M)$ two-pointer scan over sorted hit positions. An alignment pair is declared concordant if:
  $$\text{MIN\_INSERT} \ (100\text{ bp}) \le \text{TLEN} \le \text{MAX\_INSERT} \ (800\text{ bp})$$
  Up to $3$ alternative candidate pairs (`MAX_CANDIDATE_PAIRS`) are retained per read.
* **Fallback Mate Rescue:**
  When one mate yields seed hits but the opposite mate fails two-pointer concordance, a rescue candidate is projected at an empirical fixed library distance:
  $$\text{RESCUE\_OFFSET} = 380\text{ bp}, \quad \text{RESCUE\_TLEN} = 480\text{ bp}$$

---

### 4. GPU Dynamic Programming Engine (Pass 1)

* **Algorithm:** Banded Smith-Waterman-Gotoh with affine gap penalties.
* **Scoring Constants:**
  - Match: $+3$
  - Mismatch penalty: $-3$
  - Gap open penalty ($O$): $-6$
  - Gap extension penalty ($E$): $-1$
  - Negative infinity ($\text{NEG\_INF}$): $-100\,000$
  - Z-drop threshold: $100$
* **Band Geometry ($W = 32$):**
  - Sheared coordinate mapping: $k = j - i$, where $i \in [0, \text{read\_len})$ and $j \in [i, i + 32)$.
  - Recurrence vector size: $33$ elements ($32$ active band cells $+ 1$ right boundary condition). Fits within $132\text{ bytes}$ per lane, remaining strictly in GPU L1/register cache.
* **Execution Model:**
  - One GPU thread corresponds to one candidate alignment task.
  - Workgroup size: $64$ threads (`WORKGROUP_SIZE`).
* **Traceback Matrix Encoding (4 bits per cell):**
  Each cell records optimal path transitions and affine extension states packed into $4\text{ bits}$:
  - Bits $[0..1]$: Optimal direction ($1 = \text{Diagonal/Match}$, $2 = \text{Up/Insertion}$, $3 = \text{Left/Deletion}$).
  - Bit $[2]$: Insertion extension flag ($1 = \text{Extend } I \to I$, $0 = \text{Open } M \to I$).
  - Bit $[3]$: Deletion extension flag ($1 = \text{Extend } D \to D$, $0 = \text{Open } M \to D$).
  - Row packing: $32\text{ cells} \times 4\text{ bits} = 128\text{ bits} = 1 \times \text{vec4<u32>}$ per read row.
  - Task footprint: $100\text{ rows} \times 16\text{ bytes} = 1600\text{ bytes}$.

---

### 5. GPU Traceback & CIGAR Construction (Pass 2)

* **Execution Order:** Sequential pass within the same command encoder, consuming the direction matrix generated by Pass 1.
* **Tournament Winner Pruning:**
  Candidates sharing the same read family evaluate primary scores branchlessly. If an alternative candidate in the family possesses a higher score (or identical score with a lower slot index), the losing thread terminates prior to traceback, suppressing VRAM direction loads.
* **Fast-Path Identity Short-Circuit:**
  If $\text{score} = \text{read\_len} \times \text{match\_score}$ ($300$ for $100\text{ bp}$), the kernel bypasses matrix traversal, immediately writing `100M` to the output array and computing reference start via $j_{\text{start}} = j_{\text{max}} - \text{read\_len} + 1$.
* **Finite-State Machine Backtracking:**
  Traverses states $M, I, D$ using direction bits and affine extension flags.
* **CIGAR Binary Output:**
  Emits up to $32$ packed `u32` words per task (`CIGAR_CAPACITY`):
  $$\text{word} = (\text{operation\_count} \ll 2) \mid \text{opcode}$$
  Opcodes: $0 = \text{M}, \ 1 = \text{I}, \ 2 = \text{D}, \ 3 = \text{S}$.
  Empty positions are padded with `0xFFFFFFFF` (`CIGAR_SENTINEL`).

---

### 6. Pipeline Concurrency & Memory Model

* **Threading Model:** Dedicated synchronous OS threads (`std::thread`) connected via bounded crossbeam channels:
  - Reader Stage: Two parallel reader threads for paired-end streaming (R1 and R2).
  - Producer Stage: Parallel seed extraction and task staging via Rayon.
  - Aggregator Stage: Structure-of-Arrays (SoA) transposition.
  - GPU Worker Stage: Double-buffered ping-pong dispatch via `wgpu`.
  - Writer Stage: Parallel zero-allocation SAM formatting and disk I/O.
* **Structure-of-Arrays (SoA) Transposition:**
  AoS task payloads ($24 \times \text{u32}$ words) are reorganized into continuous planar columns matching the batch size, ensuring aligned, coalesced global memory access on the GPU.
* **Double Buffering (Ping-Pong):**
  Maintains two discrete execution slots (`Slot 0` and `Slot 1`) on the device. Compute pass execution for batch $N$ overlaps with host-side staging readback of batch $N-1$ and input preparation for batch $N+1$.
* **Adaptive Batch Sizing (AIMD):**
  Monitors kernel submission latency, dynamically scaling task counts per batch:
  $$\text{batch\_size} \gets \begin{cases} \min(\text{batch\_size} + 4096, \text{max\_size}) & \text{if } t_{\text{elapsed}} < 20\text{ ms} \\ \max(\lfloor\text{batch\_size} \times 0.75\rfloor \ \& \ \neg 1, \ 8192) & \text{if } t_{\text{elapsed}} > 50\text{ ms} \end{cases}$$

---

### 7. SAM Format Specification Compliance (v1.6)

* **Stream Target:** Standard output (`stdout`) by default; diverted to file via `-o <FILE>`.
* **Header Structure:** Emits `@HD VN:1.6 SO:unsorted` followed by `@SQ` lines for each contig containing chromosome names (`SN`) and lengths (`LN`).
* **Flag Conformance:** Bitwise flags strictly observe SAM specifications:
  - `0x1`: Read paired in sequencing
  - `0x2`: Proper pair (concordant coordinates and orientation)
  - `0x4`: Query unmapped
  - `0x8`: Mate unmapped
  - `0x10`: Query reverse strand
  - `0x20`: Mate reverse strand
  - `0x40`: First mate in pair (R1)
  - `0x80`: Second mate in pair (R2)
* **Strand Normalization:**
  Reads mapped to the reverse strand (`0x10`) emit reverse-complemented nucleotide sequences (`SEQ`) and reversed base quality strings (`QUAL`).
* **Automated Tail Harmonization (Query Alignment Balancing):**
  Trailing bases extending beyond the alignment window are automatically converted into soft-clips (`S`), ensuring strict mathematical balance:
  $$\sum (\text{M} + \text{I} + \text{S}) = \text{len}(\text{SEQ})$$
* **Tags:**
  Emits alignment score under the optional tag `AS:i:<score>`. Mapping quality is fixed at `60` for confident mappings and `0` for unmapped or low-scoring reads (`MIN_MAPPED_SCORE = 200`).