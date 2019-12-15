
#![no_std]
#![feature(const_generics)]

pub mod p8;
pub mod p16;
pub mod p32;
pub mod p64;
pub mod quire;

/// BREF structure: boolean, regime, exponent, fraction
/// sign
/// 	boolean
/// regime
/// 	For a n-bit posit, regime can be of length 2 to (n − 1).
/// 	All bits have the same value, opposed to the sign bit.
/// exponent
/// 	Can be 0.
/// fraction
/// 	Can be 0.
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Default, Debug)]
struct Posit<
	R: Default + Sized + Send + Sync,
	E: Default + Sized + Send + Sync,
	F: Default + Sized + Send + Sync
>(bool, R, E, F);

#[allow(dead_code)]
impl<
	R: Default + Sized + Send + Sync,
	E: Default + Sized + Send + Sync,
	F: Default + Sized + Send + Sync
> Posit<R, E, F> {
	#[inline]
	fn get_sign(&self) -> bool {
		self.0
	}

	#[inline]
	fn set_sign(&mut self, sign: bool) {
		self.0 = sign;
	}

	#[inline]
	fn set_regime(&mut self, regime: R) {
		self.1 = regime;
	}

	#[inline]
	fn set_exponent(&mut self, exponent: E) {
		self.2 = exponent;
	}

	#[inline]
	fn set_fraction(&mut self, fraction: F) {
		self.3 = fraction;
	}
}

#[allow(dead_code)]
impl<
	R: Copy + Default + Sized + Send + Sync,
	E: Copy + Default + Sized + Send + Sync,
	F: Copy + Default + Sized + Send + Sync
> Posit<R, E, F> {
	#[inline]
	fn get_regime(&self) -> R {
		self.1
	}

	#[inline]
	fn get_exponent(&self) -> E {
		self.2
	}

	#[inline]
	fn get_fraction(&self) -> F {
		self.3
	}
}

/// A classification of floating point numbers.
///
/// This `enum` is used as the return type for [`f32::classify`] and [`f64::classify`]. See
/// their documentation for more.
///
/// [`f32::classify`]: ../../std/primitive.f32.html#method.classify
/// [`f64::classify`]: ../../std/primitive.f64.html#method.classify
///
/// # Examples
///
/// ```
/// use std::num::FpCategory;
/// use std::f32;
///
/// let num = 12.4_f32;
/// let inf = f32::INFINITY;
/// let zero = 0f32;
/// let sub: f32 = 1.1754942e-38;
/// let nan = f32::NAN;
///
/// assert_eq!(num.classify(), FpCategory::Normal);
/// assert_eq!(inf.classify(), FpCategory::Infinite);
/// assert_eq!(zero.classify(), FpCategory::Zero);
/// assert_eq!(nan.classify(), FpCategory::Nan);
/// assert_eq!(sub.classify(), FpCategory::Subnormal);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PositCategory {
	/// "Not a Real", for example sqrt(-1).
	Nar,

	/// Positive or negative zero.
	Zero,

	/// A regular floating point number.
	Normal,
}

trait Trigonometry {

}
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Default)]
pub trait Posit: Trigonometry {}
pub trait Operations {}

#[cfg(test)]
mod tests {
	#[test]
	fn it_works() {
		assert_eq!(2 + 2, 4);
	}
}
