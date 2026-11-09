use std::fmt;

pub const MATCH_SCORE: i32 = 3;
pub const MISMATCH_PENALTY: i32 = -3;
pub const GAP_OPEN: i32 = -6;
pub const GAP_EXTEND: i32 = -1;
pub const NEG_INF: i32 = -100_000;
const STATE_M: u8 = 0;
const STATE_I: u8 = 1;
const STATE_D: u8 = 2;
/// Represents atomic CIGAR operations.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CigarOp {
    Match(u32),    // 'M': Alignment match or mismatch
    Insert(u32),   // 'I': Insertion into query read
    Delete(u32),   // 'D': Deletion from reference
    SoftClip(u32), // 'S': Soft-clipped query bases
}

impl fmt::Display for CigarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Match(len) => write!(f, "{len}M"),
            Self::Insert(len) => write!(f, "{len}I"),
            Self::Delete(len) => write!(f, "{len}D"),
            Self::SoftClip(len) => write!(f, "{len}S"),
        }
    }
}

/// Result of reference CPU Smith-Waterman-Gotoh alignment.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub max_score: i32,
    pub ref_start: usize,
    pub cigar: Vec<CigarOp>,
}

/// Helper struct for accumulating run-length encoded CIGAR operations.
#[derive(Default)]
struct CigarAccumulator {
    ops: Vec<CigarOp>,
    current_op: Option<u8>,
    current_count: u32,
}

impl CigarAccumulator {
    fn push(&mut self, op: u8, count: u32) {
        if self.current_op == Some(op) {
            self.current_count += count;
        } else {
            self.flush();
            self.current_op = Some(op);
            self.current_count = count;
        }
    }

    fn flush(&mut self) {
        if let Some(op) = self.current_op.take() {
            let item = match op {
                0 => CigarOp::Match(self.current_count),
                1 => CigarOp::Insert(self.current_count),
                2 => CigarOp::Delete(self.current_count),
                3 => CigarOp::SoftClip(self.current_count),
                _ => unreachable!("Invalid CIGAR opcode: {op}"),
            };
            self.ops.push(item);
            self.current_count = 0;
        }
    }

    fn finish(mut self) -> Vec<CigarOp> {
        self.flush();
        self.ops
    }
}

/// Formats a CIGAR slice into a standard SAM CIGAR string (reversed from traceback order).
pub fn format_cigar(cigar: &[CigarOp]) -> String {
    cigar.iter().rev().map(ToString::to_string).collect()
}

/// CPU reference implementation of Smith-Waterman-Gotoh with affine gap penalties.
/// Serves as the golden standard for validating GPU compute kernels.
#[allow(clippy::too_many_lines)]
fn align_sw_gotoh(read: &[u8], reference: &[u8]) -> AlignmentResult {
    let read_len = read.len();
    let ref_len = reference.len();

    let mut dp_m = vec![0i32; ref_len + 1];
    let mut dp_i = vec![NEG_INF; ref_len + 1];

    // Directions: 0 = Stop, 1 = Diagonal (M), 2 = Up (I), 3 = Left (D)
    let mut directions = vec![vec![0u8; ref_len]; read_len];

    let mut max_score = 0;
    let mut max_i = 0;
    let mut max_j = 0;

    // 1. Dynamic Programming Forward Recurrence
    for i in 0..read_len {
        let char_read = read[i];
        let mut prev_diag = 0;
        let mut left_m = 0;
        let mut left_d = NEG_INF;

        for j in 0..ref_len {
            let char_ref = reference[j];
            let match_val = if char_read == char_ref {
                MATCH_SCORE
            } else {
                MISMATCH_PENALTY
            };
            let m_diag = prev_diag + match_val;

            // Insertion (vertical: gap in reference)
            let i_open = dp_m[j + 1] + GAP_OPEN + GAP_EXTEND;
            let i_ext = dp_i[j + 1] + GAP_EXTEND;
            let i_up = i_open.max(i_ext);
            let i_is_ext = u8::from(i_ext >= i_open);

            // Deletion (horizontal: gap in query read)
            let d_open = left_m + GAP_OPEN + GAP_EXTEND;
            let d_ext = left_d + GAP_EXTEND;
            let d_left = d_open.max(d_ext);
            let d_is_ext = u8::from(d_ext >= d_open);

            // Optimal local alignment score (clamped at 0 for Smith-Waterman)
            let current = 0.max(m_diag.max(i_up.max(d_left)));

            let dir_m = if current == 0 {
                0u8
            } else if current == m_diag {
                1u8
            } else if current == i_up {
                2u8
            } else {
                3u8
            };

            // 4-bit packing: bits [0..1] = dir_m, bit [2] = i_is_ext, bit [3] = d_is_ext
            directions[i][j] = dir_m | (i_is_ext << 2) | (d_is_ext << 3);

            prev_diag = dp_m[j + 1];
            dp_m[j + 1] = current;
            dp_i[j + 1] = i_up;
            left_m = current;
            left_d = d_left;

            if current > max_score {
                max_score = current;
                max_i = i;
                max_j = j;
            }
        }
    }

    // 2. Traceback State Machine
    let mut acc = CigarAccumulator::default();

    // Right soft-clip trailing query bases
    let clip_end = u32::try_from(read_len - 1 - max_i).unwrap();
    if clip_end > 0 {
        acc.push(3, clip_end);
    }

    let mut state = STATE_M;
    let mut curr_i = i32::try_from(max_i).unwrap();
    let mut curr_j = i32::try_from(max_j).unwrap();
    let mut ref_start = max_j;

    while curr_i >= 0 && curr_j >= 0 {
        let i: usize = curr_i.try_into().unwrap();
        let j: usize = curr_j.try_into().unwrap();

        let cell = directions[i][j];
        let dir_m = cell & 0b0011;
        let i_ext = (cell >> 2) & 1;
        let d_ext = (cell >> 3) & 1;

        ref_start = curr_j.try_into().unwrap();

        match state {
            STATE_M => {
                if dir_m == 0 {
                    // Local alignment termination boundary
                    let clip_start = (curr_i + 1).cast_unsigned();
                    if clip_start > 0 {
                        acc.push(3, clip_start);
                    }
                    break;
                } else if dir_m == 1 {
                    acc.push(0, 1);
                    if curr_i == 0 || curr_j == 0 {
                        if curr_i > 0 {
                            acc.push(3, curr_i.cast_unsigned());
                        }
                        break;
                    }
                    curr_i -= 1;
                    curr_j -= 1;
                } else if dir_m == 2 {
                    acc.push(1, 1);
                    if curr_i == 0 {
                        break;
                    }
                    curr_i -= 1;
                    state = if i_ext == 1 { STATE_I } else { STATE_M };
                } else {
                    acc.push(2, 1);
                    if curr_j == 0 {
                        if curr_i > 0 {
                            acc.push(3, (curr_i + 1).cast_unsigned());
                        }
                        break;
                    }
                    curr_j -= 1;
                    state = if d_ext == 1 { STATE_D } else { STATE_M };
                }
            }
            STATE_I => {
                acc.push(1, 1);
                if curr_i == 0 {
                    break;
                }
                curr_i -= 1;
                if i_ext == 0 {
                    state = STATE_M;
                }
            }
            STATE_D => {
                acc.push(2, 1);
                if curr_j == 0 {
                    if curr_i > 0 {
                        acc.push(3, (curr_i + 1).cast_unsigned());
                    }
                    break;
                }
                curr_j -= 1;
                if d_ext == 0 {
                    state = STATE_M;
                }
            }
            _ => unreachable!(),
        }
    }

    AlignmentResult {
        max_score,
        ref_start,
        cigar: acc.finish(),
    }
}

/// Diagnostic utility: prints an ANSI-colored view of the DP matrix with the traceback path highlighted.
pub fn print_dp_matrix_window(
    read: &[u8],
    reference: &[u8],
    directions: &[Vec<u8>],
    i_range: std::ops::Range<usize>,
    j_range: std::ops::Range<usize>,
    path: &[(usize, usize)],
) {
    print!("      ");
    for j in j_range.clone() {
        print!("  {:^3}", reference[j] as char);
    }
    eprintln!();

    for i in i_range {
        print!("{:^3} {:^2} |", read[i] as char, i);
        for j in j_range.clone() {
            let dir = directions[i][j] & 0b0011;
            let symbol = match dir {
                1 => " ↖ ",
                2 => " ↑ ",
                3 => " ← ",
                _ => " · ",
            };

            if path.contains(&(i, j)) {
                print!("\x1b[1;32m{symbol:^5}\x1b[0m"); // Green highlight for path
            } else {
                print!("{symbol:^5}");
            }
        }
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_match() {
        let ref_seq = format!("{}{}{}", "C".repeat(14), "A".repeat(100), "G".repeat(14));
        let read_seq = "A".repeat(100);

        let result = align_sw_gotoh(read_seq.as_bytes(), ref_seq.as_bytes());

        assert_eq!(result.max_score, 300, "Perfect 100bp match must score 300");
        assert_eq!(format_cigar(&result.cigar), "100M");
        assert_eq!(
            result.ref_start, 14,
            "Reference offset should match flank size (14)"
        );
    }

    #[test]
    fn test_insertion_gotoh() {
        let ref_seq = format!("{}{}{}", "C".repeat(14), "A".repeat(100), "G".repeat(14));
        let read_seq = format!("{}TTTTT{}", "A".repeat(40), "A".repeat(55));

        let result = align_sw_gotoh(read_seq.as_bytes(), ref_seq.as_bytes());

        // 95 matches (285) - gap open (6) - 5 gap extensions (5) = 274
        assert_eq!(result.max_score, 274);
        assert_eq!(format_cigar(&result.cigar), "40M5I55M");
        assert_eq!(result.ref_start, 14);
    }
}
