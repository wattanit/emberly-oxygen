//! In-house safetensors reader (HC-1, Spec §5): an 8-byte little-endian
//! header length, a JSON header mapping each tensor name to dtype, shape and
//! byte range, then raw data. The file is memory-mapped and each requested
//! tensor is copied into an owned f32 tensor. Nothing in a file, however
//! hostile, may cause a panic (Spec §14).

// Nothing calls the reader until weight binding (task 3.5), which removes this.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};

use crate::{Error, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dtype {
    F32,
    Bf16,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
        }
    }
}

#[derive(Debug)]
struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    /// Byte range within the data region (after the header).
    start: usize,
    end: usize,
}

/// An opened safetensors file: header validated, data still on disk.
pub struct SafeTensors {
    path: PathBuf,
    map: Mmap,
    data_start: usize,
    entries: BTreeMap<String, Entry>,
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// The header as an ordered list, so a repeated name is seen rather than
/// silently overwritten by the JSON map.
struct RawHeader(Vec<(String, serde_json::Value)>);

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawHeader;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<RawHeader, A::Error> {
                let mut items = Vec::new();
                while let Some(kv) = m.next_entry()? {
                    items.push(kv);
                }
                Ok(RawHeader(items))
            }
        }
        d.deserialize_map(V)
    }
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, Error> {
        let malformed = |reason: String| Error::Malformed {
            path: path.to_owned(),
            reason,
        };
        let io = |source| Error::Io {
            path: path.to_owned(),
            source,
        };

        // Sharded checkpoints name their pieces in an index file beside the weights.
        if let Some(dir) = path.parent() {
            let index = dir.join("model.safetensors.index.json");
            if index.exists() {
                return Err(Error::Unsupported {
                    component: "weights layout".into(),
                    value: format!("sharded weights ({} exists); only a single file is supported", index.display()),
                });
            }
        }

        let file = File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        if len < 8 {
            return Err(malformed(format!("{len} bytes is too short for a safetensors file")));
        }
        // SAFETY: the map is read-only and the file is a model the user chose; a
        // concurrent truncation by another process is the usual mmap caveat.
        let map = unsafe { Mmap::map(&file) }.map_err(io)?;

        let header_len = u64::from_le_bytes(map[..8].try_into().expect("8 bytes"));
        let data_start = usize::try_from(header_len)
            .ok()
            .and_then(|n| n.checked_add(8))
            .filter(|&end| end <= map.len())
            .ok_or_else(|| {
                malformed(format!(
                    "header length {header_len} does not fit in a file of {} bytes",
                    map.len()
                ))
            })?;
        let data_len = map.len() - data_start;

        let raw: RawHeader = serde_json::from_slice(&map[8..data_start])
            .map_err(|e| malformed(format!("header is not a JSON object: {e}")))?;

        let mut entries = BTreeMap::new();
        for (name, value) in raw.0 {
            if name == "__metadata__" {
                continue;
            }
            let bad = |reason: String| malformed(format!("tensor {name}: {reason}"));
            let e: RawEntry = serde_json::from_value(value)
                .map_err(|e| bad(format!("bad header entry: {e}")))?;
            let dtype = match e.dtype.as_str() {
                "F32" => Dtype::F32,
                "BF16" => Dtype::Bf16,
                other => {
                    return Err(Error::Unsupported {
                        component: format!("dtype of tensor {name}"),
                        value: other.into(),
                    });
                }
            };
            let [start, end] = e.data_offsets;
            if start > end || end > data_len {
                return Err(bad(format!(
                    "byte range {start}..{end} is outside the {data_len}-byte data region"
                )));
            }
            let elements = e.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
            let bytes = elements.and_then(|n| n.checked_mul(dtype.size()));
            if bytes != Some(end - start) {
                return Err(bad(format!(
                    "shape {:?} of {dtype:?} does not match the {}-byte range",
                    e.shape,
                    end - start
                )));
            }
            let entry = Entry { dtype, shape: e.shape, start, end };
            if entries.insert(name.clone(), entry).is_some() {
                return Err(bad("listed twice in the header".into()));
            }
        }

        // Empty tensors occupy no bytes and cannot overlap anything.
        let mut ranges: Vec<(&String, &Entry)> =
            entries.iter().filter(|(_, e)| e.end > e.start).collect();
        ranges.sort_by_key(|(_, e)| e.start);
        let mut furthest: Option<(&String, usize)> = None;
        for (name, e) in ranges {
            if let Some((prev, end)) = furthest {
                if e.start < end {
                    return Err(malformed(format!("tensors {prev} and {name} overlap")));
                }
            }
            furthest = Some((name, e.end));
        }

        Ok(SafeTensors { path: path.to_owned(), map, data_start, entries })
    }

    /// Tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Copies one tensor out as f32; BF16 is widened, which is exact.
    pub fn tensor(&self, name: &str) -> Result<Tensor, Error> {
        let entry = self.entries.get(name).ok_or_else(|| Error::TensorMismatch {
            name: name.into(),
            reason: format!("not in {}", self.path.display()),
        })?;
        let bytes = &self.map[self.data_start + entry.start..self.data_start + entry.end];
        let data = match entry.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| widen_bf16(u16::from_le_bytes(b.try_into().expect("2 bytes"))))
                .collect(),
        };
        Tensor::new(entry.shape.clone(), data).map_err(|e| match e {
            Error::TensorMismatch { reason, .. } => Error::TensorMismatch {
                name: name.into(),
                reason,
            },
            other => other,
        })
    }
}

/// bfloat16 is the top 16 bits of an f32, so widening is a shift and exact.
fn widen_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Writes `bytes` to a fresh file in its own temp directory and returns the path.
    fn temp_file(bytes: &[u8]) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "oxygen-st-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Builds a safetensors file image from a header JSON string and a data region.
    fn image(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    fn open(bytes: &[u8]) -> Result<SafeTensors, Error> {
        SafeTensors::open(&temp_file(bytes))
    }

    fn reject(bytes: &[u8]) -> String {
        open(bytes).err().expect("expected rejection").to_string()
    }

    fn f32_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    // FR-11, Spec §5: F32 data is copied as stored; the metadata entry is ignored.
    #[test]
    fn reads_f32_tensors_and_ignores_metadata() {
        let header = r#"{"__metadata__":{"format":"pt"},
            "a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
            "b":{"dtype":"F32","shape":[3],"data_offsets":[16,28]}}"#;
        let data = f32_bytes(&[1., 2., 3., 4., 5., 6., 7.]);
        let st = open(&image(header, &data)).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(st.shape("a"), Some(&[2, 2][..]));
        let a = st.tensor("a").unwrap();
        assert_eq!((a.shape(), a.data()), (&[2, 2][..], &[1., 2., 3., 4.][..]));
        assert_eq!(st.tensor("b").unwrap().data(), &[5., 6., 7.]);
        assert!(matches!(st.tensor("c"), Err(Error::TensorMismatch { .. })));
    }

    // HC-1, Spec §5: BF16 widening is exact, for every one of the 65536 patterns.
    #[test]
    fn bf16_widening_is_exact_for_every_pattern() {
        for bits in 0..=u16::MAX {
            let wide = widen_bf16(bits);
            assert_eq!(wide.to_bits(), u32::from(bits) << 16, "{bits:#06x}");
        }
        assert_eq!(widen_bf16(0x3F80), 1.0);
        assert_eq!(widen_bf16(0xC000), -2.0);
        assert_eq!(widen_bf16(0x0000).to_bits(), 0);
        assert_eq!(widen_bf16(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(widen_bf16(0x7F80), f32::INFINITY);
        assert!(widen_bf16(0x7FC0).is_nan());
        assert_eq!(widen_bf16(0x0001), 2f32.powi(-133));
    }

    #[test]
    fn reads_bf16_tensors_through_the_file() {
        let header = r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let data: Vec<u8> = [0x3F80u16, 0xC000, 0x3F00, 0x0000]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let t = open(&image(header, &data)).unwrap().tensor("w").unwrap();
        assert_eq!(t.data(), &[1.0, -2.0, 0.5, 0.0]);
    }

    // FR-12, Spec §5: other dtypes are rejected naming the tensor and the dtype.
    #[test]
    fn other_dtypes_are_rejected_by_name() {
        let header = r#"{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let msg = reject(&image(header, &[0, 0]));
        assert!(msg.contains("tensor w") && msg.contains("F16"), "{msg}");
    }

    // FR-12, Spec §5: sharded weights are rejected with an error.
    #[test]
    fn sharded_weights_are_rejected() {
        let path = temp_file(&image("{}", &[]));
        std::fs::write(path.with_file_name("model.safetensors.index.json"), "{}").unwrap();
        let err = SafeTensors::open(&path).err().expect("expected rejection");
        assert!(matches!(err, Error::Unsupported { .. }));
        assert!(err.to_string().contains("sharded"), "{err}");
    }

    // Spec §14: hostile files are errors, never panics.
    #[test]
    fn truncated_and_hostile_files_are_malformed() {
        for bytes in [
            &b""[..],
            &b"\x01\x02\x03"[..],
            &u64::MAX.to_le_bytes()[..],
            &image("{}", &[])[..9],
        ] {
            let err = open(bytes).err().expect("expected rejection");
            assert!(matches!(err, Error::Malformed { .. }), "{err}");
        }
        // Header length that overflows when 8 is added.
        let mut huge = (u64::MAX - 4).to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(matches!(open(&huge), Err(Error::Malformed { .. })));
        // Header that is not JSON, or JSON that is not an object.
        for header in ["not json", "[]", "{\"a\":1}"] {
            let msg = reject(&image(header, &[]));
            assert!(msg.contains("malformed"), "{header}: {msg}");
        }
    }

    #[test]
    fn bad_ranges_and_shapes_name_the_tensor() {
        let cases = [
            // Range past the end of the data region.
            r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}"#,
            // Start after end.
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
            // Shape disagrees with the byte length.
            r#"{"t":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            // Shape whose element count overflows.
            r#"{"t":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
            // Missing field.
            r#"{"t":{"dtype":"F32","shape":[2]}}"#,
        ];
        for header in cases {
            let msg = reject(&image(header, &[0; 8]));
            assert!(msg.contains("tensor t"), "{header}: {msg}");
        }
    }

    #[test]
    fn overlapping_and_duplicate_entries_are_rejected() {
        let overlap = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let msg = reject(&image(overlap, &[0; 12]));
        assert!(msg.contains("overlap") && msg.contains('a') && msg.contains('b'), "{msg}");

        let dup = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let msg = reject(&image(dup, &[0; 8]));
        assert!(msg.contains("tensor a") && msg.contains("twice"), "{msg}");
    }

    #[test]
    fn an_empty_tensor_does_not_hide_an_overlap() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "e":{"dtype":"F32","shape":[0],"data_offsets":[4,4]},
            "b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let msg = reject(&image(header, &[0; 8]));
        assert!(msg.contains("overlap"), "{msg}");
    }

    #[test]
    fn empty_tensors_do_not_count_as_overlap() {
        let header = r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
            "b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let st = open(&image(header, &[0; 4])).unwrap();
        assert!(st.tensor("a").unwrap().data().is_empty());
    }

    // FR-11: the real published file opens and lists its tensors.
    #[test]
    fn real_model_file_opens() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m/model.safetensors");
        assert!(
            path.exists(),
            "{}: missing; run `cd oracle && uv run python download.py`",
            path.display()
        );
        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.names().count(), 272);
        assert_eq!(st.shape("model.embed_tokens.weight"), Some(&[49152, 576][..]));
        assert_eq!(st.shape("model.norm.weight"), Some(&[576][..]));
        let norm = st.tensor("model.norm.weight").unwrap();
        assert!(norm.data().iter().all(|x| x.is_finite()));
    }
}
