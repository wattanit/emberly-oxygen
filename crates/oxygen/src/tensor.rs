use crate::Error;

/// A contiguous `f32` array with a row-major shape (Spec §3). Loading is all
/// it supports for now; operations arrive with the kernels in Phase 4.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Fails if `data` does not hold exactly the number of elements `shape` implies.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor, Error> {
        let expected = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if expected != Some(data.len()) {
            return Err(Error::TensorMismatch {
                name: String::new(),
                reason: format!("shape {shape:?} does not match {} elements", data.len()),
            });
        }
        Ok(Tensor { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Row `i` of a matrix, a contiguous slice: for a published `[out, in]`
    /// weight this is the `in` values one output element dots with.
    /// Panics if the tensor is not 2-D or `i` is out of range (internal invariant).
    pub fn row(&self, i: usize) -> &[f32] {
        assert_eq!(self.shape.len(), 2, "row() needs a 2-D tensor, got {:?}", self.shape);
        let cols = self.shape[1];
        &self.data[i * cols..(i + 1) * cols]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // HC-1, Spec §3: shape and data are kept as given, row-major.
    #[test]
    fn construction_keeps_shape_and_data() {
        let t = Tensor::new(vec![2, 3], vec![0., 1., 2., 3., 4., 5.]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), &[0., 1., 2., 3., 4., 5.]);
    }

    // HC-1, Spec §3: scalar and zero-length shapes are valid tensors.
    #[test]
    fn scalar_and_empty_shapes() {
        assert!(Tensor::new(vec![], vec![7.]).is_ok());
        assert!(Tensor::new(vec![0, 4], vec![]).is_ok());
    }

    // Spec §14: a length that disagrees with the shape is an error, never a panic.
    #[test]
    fn length_shape_mismatch_is_rejected() {
        assert!(matches!(
            Tensor::new(vec![2, 3], vec![0.; 5]),
            Err(Error::TensorMismatch { .. })
        ));
        assert!(Tensor::new(vec![2, 3], vec![0.; 7]).is_err());
        assert!(Tensor::new(vec![], vec![]).is_err());
        // A shape whose element count overflows must not wrap around to a match.
        assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
    }

    // Spec §3: each row of an `[out, in]` matrix is one contiguous slice.
    #[test]
    fn rows_are_contiguous_slices() {
        let t = Tensor::new(vec![3, 2], vec![0., 1., 2., 3., 4., 5.]).unwrap();
        assert_eq!(t.row(0), &[0., 1.]);
        assert_eq!(t.row(2), &[4., 5.]);
    }
}
