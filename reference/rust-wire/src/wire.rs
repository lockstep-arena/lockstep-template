//! A HAND-WRITTEN Lockstep-wire (v1) reader/writer — the whole point.
//!
//! The wire is a *spec*, not a library: this file re-implements it from the
//! published document (`docs/wire.md` in this template — the vendored copy
//! of the platform's normative spec) in a few hundred lines, with no
//! dependency on the reference crate. If you are porting an agent to Go,
//! Zig or C, this file is the shape of what you will write. The goldens
//! under `tests/fixtures/` (published with the spec) pin it: `cargo test`
//! decodes and re-encodes them byte-for-byte.
//!
//! Encoding rules (the short version — the spec is normative):
//! - everything little-endian; `f32` is IEEE-754 binary32
//! - `str` = `u16` length + UTF-8 bytes, no terminator
//! - every value, slice and column carries a `doc` string (slices and
//!   columns a `unit` too), and the seat-init ends with the goal / reward /
//!   ends brief and the declared metrics — the environment documents
//!   itself; decoders that don't care skip the strings
//! - a value's bytes are row-major, `dtype size × product(shape)` exactly
//! - `dtype`: 0 = f32 (4 B), 1 = u8 (1 B), 2 = i32 (4 B)
//! - values appear in DECLARED order with exact byte lengths
//! - every `ValueSpec` is a length-prefixed region: read the fields you
//!   know, then skip to the end of the region (a later revision may append
//!   fields there); a region that ends before the known fields is an error
//! - the seat-init is tail-extensible: after `ends` comes the `metrics`
//!   section, and a reader ignores any bytes after the sections it knows.
//!   No bytes at all after `ends` means `metrics = []`; a tail that starts
//!   and stops short is an error. `View` and `Input` stay strict.

#![allow(dead_code)]

// ── primitives ──────────────────────────────────────────────────────────

pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[derive(Debug)]
pub struct WireError;

type Result<T> = core::result::Result<T, WireError>;

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(WireError)?;
        let s = self.data.get(self.pos..end).ok_or(WireError)?;
        self.pos = end;
        Ok(s)
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn magic(&mut self, expected: &[u8; 4]) -> Result<()> {
        (self.take(4)? == expected).then_some(()).ok_or(WireError)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn str_(&mut self) -> Result<String> {
        let len = self.u16()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| WireError)
    }
}

// ── declarations (SeatInit) ─────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    U8,
    I32,
}

impl Dtype {
    fn parse(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::F32),
            1 => Ok(Self::U8),
            2 => Ok(Self::I32),
            _ => Err(WireError),
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 | Self::I32 => 4,
        }
    }
}

/// A named run of elements inside a rank-1 value — documentation, never a
/// second layout.
#[derive(Clone, Debug)]
pub struct Slice {
    pub name: String,
    /// What this run of elements IS, in the engine's words.
    pub doc: String,
    /// Physical unit (`"m"`, `"rad/s"`); empty when dimensionless. `"code"`
    /// marks an integer category whose table is in `doc`.
    pub unit: String,
    pub start: u32,
    pub len: u32,
}

/// One documented column of the LAST axis of a rank ≥ 2 value — the
/// per-column twin of a [`Slice`]. A `history f32[16, 8]` declares eight,
/// in column order; a rank ≤ 1 value declares none.
#[derive(Clone, Debug)]
pub struct ColumnSpec {
    pub name: String,
    /// What this column IS (and the code table when `unit` is `"code"`).
    pub doc: String,
    /// As for a slice.
    pub unit: String,
}

#[derive(Clone, Debug)]
pub struct ValueSpec {
    pub name: String,
    /// What this value is (and, for rank ≥ 2, what the rows mean and in
    /// what order).
    pub doc: String,
    pub dtype: Dtype,
    pub shape: Vec<u32>,
    pub low: f32,
    pub high: f32,
    /// Per-element bounds (`numel` each) when declared.
    pub elem_bounds: Option<(Vec<f32>, Vec<f32>)>,
    /// Documented runs of a rank-1 value.
    pub slices: Vec<Slice>,
    /// Documented columns of the last axis of a rank ≥ 2 value: empty, or
    /// exactly `shape[rank-1]` entries.
    pub columns: Vec<ColumnSpec>,
}

impl ValueSpec {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1)
    }

    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    pub fn slice(&self, name: &str) -> Option<&Slice> {
        self.slices.iter().find(|s| s.name == name)
    }

    /// A documented column by name (rank ≥ 2 values).
    pub fn column(&self, name: &str) -> Option<&ColumnSpec> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The index along the last axis of the column named `name` — what you
    /// add to `row * shape[rank-1]` to reach it in the flat elements.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Element bounds at `i`: per-element when declared, else the scalars.
    pub fn bounds_at(&self, i: usize) -> (f32, f32) {
        match &self.elem_bounds {
            Some((low, high)) => (low[i], high[i]),
            None => (self.low, self.high),
        }
    }

    /// The wire's own neutral: the midpoint of finite bounds, else 0.
    pub fn neutral_f32(&self) -> Vec<f32> {
        (0..self.numel())
            .map(|i| {
                let (low, high) = self.bounds_at(i);
                if low.is_finite() && high.is_finite() {
                    (low + high) / 2.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// One length-prefixed region: `len u32`, then the fields below. The
    /// region is taken whole, the known fields are read out of it, and
    /// whatever the region holds after them is skipped — a later revision's
    /// fields. A region that ends before the known fields (or a `len` that
    /// runs past the input) is an error.
    fn parse(outer: &mut Reader) -> Result<Self> {
        let len = outer.u32()? as usize;
        let mut r = Reader::new(outer.take(len)?);
        let name = r.str_()?;
        let doc = r.str_()?;
        let dtype = Dtype::parse(r.u8()?)?;
        let rank = r.u8()? as usize;
        let shape: Vec<u32> = (0..rank).map(|_| r.u32()).collect::<Result<_>>()?;
        let low = r.f32()?;
        let high = r.f32()?;
        let numel = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
        let elem_bounds = if r.u8()? != 0 {
            // Bounds-check before allocating: a hostile numel must not OOM.
            if numel.checked_mul(8).ok_or(WireError)? > r.remaining() {
                return Err(WireError);
            }
            let lo: Vec<f32> = (0..numel).map(|_| r.f32()).collect::<Result<_>>()?;
            let hi: Vec<f32> = (0..numel).map(|_| r.f32()).collect::<Result<_>>()?;
            Some((lo, hi))
        } else {
            None
        };
        let n_slices = r.u32()? as usize;
        let slices = (0..n_slices)
            .map(|_| {
                Ok(Slice {
                    name: r.str_()?,
                    doc: r.str_()?,
                    unit: r.str_()?,
                    start: r.u32()?,
                    len: r.u32()?,
                })
            })
            .collect::<Result<_>>()?;
        let n_columns = r.u32()? as usize;
        let columns: Vec<ColumnSpec> = (0..n_columns)
            .map(|_| {
                Ok(ColumnSpec {
                    name: r.str_()?,
                    doc: r.str_()?,
                    unit: r.str_()?,
                })
            })
            .collect::<Result<_>>()?;
        // Columns are exactly one per element of the last axis, or none —
        // and only a rank ≥ 2 value has them.
        if !columns.is_empty() {
            let expected = if rank >= 2 {
                shape[rank - 1] as usize
            } else {
                0
            };
            if columns.len() != expected {
                return Err(WireError);
            }
        }
        // Anything left in the region belongs to a later revision: skipped
        // (the outer reader already stands at the end of the region).
        Ok(Self {
            name,
            doc,
            dtype,
            shape,
            low,
            high,
            elem_bounds,
            slices,
            columns,
        })
    }
}

/// The seat's brief — goal, reward, episode end — in the engine's words.
/// Print it when your agent starts: it is the environment's own README.
#[derive(Debug, Default)]
pub struct Brief {
    pub goal: String,
    pub reward: String,
    pub ends: String,
}

/// What a session metric MEANS (the wire byte is the discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetricKind {
    /// The 0–100 episode score, or another score-like number.
    Score = 0,
    /// An integer count of things.
    Count = 1,
    /// A fraction or percentage.
    Rate = 2,
    /// A duration, in the unit given.
    Duration = 3,
    /// Money, in the unit given.
    Money = 4,
    /// A 0/1 flag (`success`, `fail-<reason>`).
    Flag = 5,
    /// A fact about the seed, shown to employers only; its key starts with
    /// `scenario-`.
    Scenario = 6,
}

impl MetricKind {
    fn parse(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Score),
            1 => Ok(Self::Count),
            2 => Ok(Self::Rate),
            3 => Ok(Self::Duration),
            4 => Ok(Self::Money),
            5 => Ok(Self::Flag),
            6 => Ok(Self::Scenario),
            _ => Err(WireError),
        }
    }
}

/// Whether a bigger value of a metric is better (the wire byte is the
/// discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetricDirection {
    Higher = 0,
    Lower = 1,
    /// A fact, not a target.
    Neutral = 2,
}

impl MetricDirection {
    fn parse(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Higher),
            1 => Ok(Self::Lower),
            2 => Ok(Self::Neutral),
            _ => Err(WireError),
        }
    }
}

/// One documented session metric: the key the engine emits at the end of
/// an episode, what it means, its unit, which way is better, a display
/// range for charts (`±inf` = fit to the data), and whether it belongs on
/// the report's headline cards.
#[derive(Clone, Debug)]
pub struct MetricSpec {
    pub key: String,
    pub doc: String,
    pub unit: String,
    pub kind: MetricKind,
    pub direction: MetricDirection,
    pub low: f32,
    pub high: f32,
    pub headline: bool,
}

impl MetricSpec {
    fn parse(r: &mut Reader) -> Result<Self> {
        Ok(Self {
            key: r.str_()?,
            doc: r.str_()?,
            unit: r.str_()?,
            kind: MetricKind::parse(r.u8()?)?,
            direction: MetricDirection::parse(r.u8()?)?,
            low: r.f32()?,
            high: r.f32()?,
            headline: r.u8()? != 0,
        })
    }
}

#[derive(Debug)]
pub struct SeatInit {
    pub seat: u32,
    pub obs: Vec<ValueSpec>,
    pub actions: Vec<ValueSpec>,
    pub meta: Vec<(String, String)>,
    pub brief: Brief,
    /// The session metrics this engine reports, documented — the first
    /// tail section. Empty when the declaration carries no tail.
    pub metrics: Vec<MetricSpec>,
}

impl SeatInit {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.magic(b"LSTI")?;
        if r.u32()? != 1 {
            return Err(WireError); // wire version we don't speak
        }
        let seat = r.u32()?;
        let n_obs = r.u32()? as usize;
        let obs = (0..n_obs)
            .map(|_| ValueSpec::parse(&mut r))
            .collect::<Result<_>>()?;
        let n_act = r.u32()? as usize;
        let actions = (0..n_act)
            .map(|_| ValueSpec::parse(&mut r))
            .collect::<Result<_>>()?;
        let n_meta = r.u32()? as usize;
        let meta = (0..n_meta)
            .map(|_| Ok((r.str_()?, r.str_()?)))
            .collect::<Result<_>>()?;
        let brief = Brief {
            goal: r.str_()?,
            reward: r.str_()?,
            ends: r.str_()?,
        };
        // The tail. Nothing after `ends` = a declaration written before the
        // metrics section existed: `metrics = []`. A tail that starts is
        // read whole — a count with too few specs behind it is an error.
        // Bytes after the sections this reader knows are IGNORED: a later
        // revision may append sections. Views and inputs stay strict.
        let metrics = if r.remaining() > 0 {
            let n_metrics = r.u32()? as usize;
            (0..n_metrics)
                .map(|_| MetricSpec::parse(&mut r))
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            seat,
            obs,
            actions,
            meta,
            brief,
            metrics,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn obs(&self, name: &str) -> Option<&ValueSpec> {
        self.obs.iter().find(|s| s.name == name)
    }

    pub fn action(&self, name: &str) -> Option<&ValueSpec> {
        self.actions.iter().find(|s| s.name == name)
    }

    /// The declared spec for a metric key.
    pub fn metric_spec(&self, key: &str) -> Option<&MetricSpec> {
        self.metrics.iter().find(|m| m.key == key)
    }

    /// The metrics marked `headline`, in declaration order.
    pub fn headline_metrics(&self) -> impl Iterator<Item = &MetricSpec> {
        self.metrics.iter().filter(|m| m.headline)
    }
}

// ── per-tick messages ───────────────────────────────────────────────────

#[derive(Debug)]
pub struct View<'a> {
    pub tick: u32,
    pub reward: f32,
    pub done: bool,
    /// Raw observation bytes in DECLARED order — sliced zero-copy from the
    /// received buffer (a big image strip is never copied).
    pub values: Vec<&'a [u8]>,
}

impl<'a> View<'a> {
    pub fn decode(bytes: &'a [u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.magic(b"LSTV")?;
        let tick = r.u32()?;
        let reward = r.f32()?;
        let done = r.u8()? != 0;
        r.take(3)?; // pad
        let n = r.u32()? as usize;
        let values = (0..n)
            .map(|_| {
                let len = r.u32()? as usize;
                r.take(len)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tick,
            reward,
            done,
            values,
        })
    }
}

/// Encode an `Input`: one raw byte blob per declared action, in order.
pub fn encode_input(values: &[Vec<u8>]) -> Vec<u8> {
    let payload: usize = values.iter().map(|t| 4 + t.len()).sum();
    let mut out = Vec::with_capacity(8 + payload);
    out.extend_from_slice(b"LSTA");
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for t in values {
        out.extend_from_slice(&(t.len() as u32).to_le_bytes());
        out.extend_from_slice(t);
    }
    out
}

/// f32 values -> raw element bytes.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The neutral raw bytes for ANY action spec — what "do nothing" means.
/// `u8`/`i32` round the midpoint (the spec's rule), so an i32 with bounds
/// 0..=3 is neutral at 2, matching the reference implementation.
pub fn neutral_bytes(spec: &ValueSpec) -> Vec<u8> {
    match spec.dtype {
        Dtype::F32 => f32_bytes(&spec.neutral_f32()),
        Dtype::I32 => spec
            .neutral_f32()
            .iter()
            .flat_map(|v| (v.round() as i32).to_le_bytes())
            .collect(),
        Dtype::U8 => spec.neutral_f32().iter().map(|v| v.round() as u8).collect(),
    }
}
