//! A HAND-WRITTEN Lockstep-wire (v1) reader/writer — the whole point.
//!
//! The wire is a *spec*, not a library: this file re-implements it from the
//! published document (`docs/wire.md` in this template — the vendored copy
//! of the platform's normative spec) in ~200 lines, with no dependency on the reference crate. If you
//! are porting an agent to Go, Zig or C, this file is the shape of what you
//! will write. The goldens under `tests/fixtures/` (published with the
//! spec) pin it: `cargo test` decodes and re-encodes them byte-for-byte.
//!
//! Encoding rules (the short version — the spec is normative):
//! - everything little-endian; `f32` is IEEE-754 binary32
//! - `str` = `u16` length + UTF-8 bytes, no terminator
//! - every value and slice carries a `doc` string (slices a `unit` too),
//!   and the seat-init ends with the goal / reward / ends brief — the
//!   environment documents itself; decoders that don't care skip the strings
//! - a value's bytes are row-major, `dtype size × product(shape)` exactly
//! - `dtype`: 0 = f32 (4 B), 1 = u8 (1 B), 2 = i32 (4 B)
//! - values appear in DECLARED order with exact byte lengths

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

#[derive(Clone, Debug)]
pub struct Slice {
    pub name: String,
    /// What this run of elements IS, in the engine's words.
    pub doc: String,
    /// Physical unit (`"m"`, `"rad/s"`); empty when dimensionless.
    pub unit: String,
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Debug)]
pub struct ValueSpec {
    pub name: String,
    /// What this value is (and, for rank ≥ 2, what rows/columns mean).
    pub doc: String,
    pub dtype: Dtype,
    pub shape: Vec<u32>,
    pub low: f32,
    pub high: f32,
    /// Per-element bounds (`numel` each) when declared.
    pub elem_bounds: Option<(Vec<f32>, Vec<f32>)>,
    pub slices: Vec<Slice>,
}

impl ValueSpec {
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

    fn parse(r: &mut Reader) -> Result<Self> {
        let name = r.str_()?;
        let doc = r.str_()?;
        let dtype = Dtype::parse(r.u8()?)?;
        let rank = r.u8()? as usize;
        let shape: Vec<u32> = (0..rank).map(|_| r.u32()).collect::<Result<_>>()?;
        let low = r.f32()?;
        let high = r.f32()?;
        let numel = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
        let elem_bounds = if r.u8()? != 0 {
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
        Ok(Self {
            name,
            doc,
            dtype,
            shape,
            low,
            high,
            elem_bounds,
            slices,
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

#[derive(Debug)]
pub struct SeatInit {
    pub seat: u32,
    pub obs: Vec<ValueSpec>,
    pub actions: Vec<ValueSpec>,
    pub meta: Vec<(String, String)>,
    pub brief: Brief,
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
        Ok(Self {
            seat,
            obs,
            actions,
            meta,
            brief,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn action(&self, name: &str) -> Option<&ValueSpec> {
        self.actions.iter().find(|s| s.name == name)
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
pub fn neutral_bytes(spec: &ValueSpec) -> Vec<u8> {
    match spec.dtype {
        Dtype::F32 => f32_bytes(&spec.neutral_f32()),
        Dtype::I32 => spec
            .neutral_f32()
            .iter()
            .flat_map(|v| (*v as i32).to_le_bytes())
            .collect(),
        Dtype::U8 => spec.neutral_f32().iter().map(|v| *v as u8).collect(),
    }
}
