
#![allow(dead_code)]

use crate::{PositCategory, Posit};
use core::{convert::From, mem, ops::*};

/*
TODO:
	pub const fn count_ones(self) -> u8;
	pub const fn count_zeros(self) -> u8;
*/

/// Max fraction (significand) size.
pub const FS: usize = 6;

/// The maximum number of bits 0, 1, 2, 3, . . . that are available for expressing the exponent.
pub const ES: usize = 0;

/// The precision of a posit format, the total number of bits (8, 16, 32, or 64).
pub const NBITS: usize = 8;

/// The smallest nonzero value expressible as a posit.
/// 2^{\frac_1_8 nbits(2 − nbits)}
/// 0,015625
pub const MINPOS: p8 = p8(0b0000_0001); // EPSILON ?

/// The largest real value expressible as a posit.
/// 2^{\frac_1_8 nbits(nbits - 2)}
pub const MAXPOS: u8 = 64;

/// The largest consecutive integer expressible as a posit.
pub const PINTMAX: u8 = 8;

/// A fixed-point format capable of storing sums of products of posits without rounding, up to some largenumber of such
/// products.
pub const QUIRE: usize = 32;

/// Exact sum quire limit.
pub const SUM_QUIRE_LIMIT: u16 = 32767;

/// Exact dot product quire limit.
pub const DOT_PRODUCT_QUIRE_LIMIT: u8 = 127;

pub const ZERO: p8 = p8(0b0000_0000);

pub const NAR: p8 = p8(0b1000_0000);

pub const USEED: u8 = 0;

/// Basic mathematical constants.
pub mod consts {
	use crate::p8::p8;

	/// Archimedes' constant (π)
	/// Real value : 3.125
	/// Standard library : 3.14159265358979323846264338327950288_f32
	pub const PI: p8 = p8(0b0110_1001);

	/// π/2
	/// Real value : 1.5625
	/// Standard library : 1.57079632679489661923132169163975144_f32
	pub const FRAC_PI_2: p8 = p8(0b0101_0010);

	/// π/3
	/// Real value : 1.0625
	/// Standard library : 1.04719755119659774615421446109316763_f32
	pub const FRAC_PI_3: p8 = p8(0b0100_0010);

	/// π/4
	/// Real value : 0.78125
	/// Standard library : 0.785398163397448309615660845819875721_f32
	pub const FRAC_PI_4: p8 = p8(0b0011_0010);

	/// π/6
	/// Real value : 0.53125
	/// Standard library : 0.52359877559829887307710723054658381_f32
	pub const FRAC_PI_6: p8 = p8(0b0010_0010);

	/// π/8
	/// Real value : 0.390625
	/// Standard library : 0.39269908169872415480783042290993786_f32
	pub const FRAC_PI_8: p8 = p8(0b0001_1001);

	/// 1/π
	/// Real value : 0.3125
	/// Standard library : 0.318309886183790671537767526745028724_f32
	pub const FRAC_1_PI: p8 = p8(0b0001_0100);

	/// 2/π
	/// Real value : 0.640625
	/// Standard library : 0.636619772367581343075535053490057448_f32
	pub const FRAC_2_PI: p8 = p8(0b0010_1001);

	/// 2/sqrt(π)
	/// Real value : 1.125
	/// Standard library : 1.12837916709551257389615890312154517_f32
	pub const FRAC_2_SQRT_PI: p8 = p8(0b0100_0100);

	/// sqrt(2)
	/// Real value : 1.40625
	/// Standard library : 1.41421356237309504880168872420969808_f32
	pub const SQRT_2: p8 = p8(0b0100_1101);

	/// 1/sqrt(2)
	/// Real value : 0.703125
	/// Standard library : 0.707106781186547524400844362104849039_f32
	pub const FRAC_1_SQRT_2: p8 = p8(0b0010_1101);

	/// Euler's number (e)
	/// Real value : 2.75
	/// Standard library : 2.71828182845904523536028747135266250_f32
	pub const E: p8 = p8(0b0110_0110);

	/// log<sub>2</sub>(e)
	/// Real value : 1.4375
	/// Standard library : 1.44269504088896340735992468100189214_f32
	pub const LOG2_E: p8 = p8(0b0100_1110);

	/// log<sub>2</sub>(10)
	/// Real value : 3.375
	/// Standard library : 3.32192809488736234787031942948939018_f32
	pub const LOG2_10: p8 = p8(0b0110_1011);

	/// log<sub>10</sub>(e)
	/// Real value : 0.4375
	/// Standard library : 0.434294481903251827651128918916605082_f32
	pub const LOG10_E: p8 = p8(0b0001_1100);

	/// log<sub>10</sub>(2)
	/// Real value : 0.296875
	/// Standard library : 0.301029995663981195213738894724493027_f32
	pub const LOG10_2: p8 = p8(0b0001_0011);

	/// ln(2)
	/// Real value : 0.6875
	/// Standard library : 0.693147180559945309417232121458176568_f32
	pub const LN_2: p8 = p8(0b0010_1100);

	/// ln(10)
	/// Real value : 2.25
	/// Standard library : 2.30258509299404568401799145468436421_f32
	pub const LN_10: p8 = p8(0b0110_0010);
}

/// Posits can exactly express all integers i in a range −pintmax ≤ i ≤ pintmax; outside that range, integers exist
/// that cannot be expressed as a posit without rounding to a different integer.
/// SRRRRRRF
#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Default)]
pub struct p8(i8);

impl p8 {
	pub fn new() -> Self { ZERO }

	/// Returns 1 if the value of posit is positive, and -1 if the value of posit is 1.
	/// If posit is zero or NaR, sign(posit) returns 0.
	///
	/// # Examples
	///
	/// ```
	/// use positrion::p8;
	///
	/// let f = p8::from(3.5_f32);
	///
	/// assert_eq!(f.signum(), 1.0);
	/// assert_eq!(p8::NAR.signum(), -1.0);
	///
	/// assert!(p8::NAR.signum().is_nar());
	/// ```
	#[inline]
	pub fn signum(self) -> i8 {
		if self == NAR || self == ZERO {
			0
		} else if self.is_sign_negative() {
			-1
		} else {
			1
		}
	}

	/// Returns `true` if this value is `NaR`.
	///
	/// ```
	/// use positron::p8;
	///
	/// let nar = p8::NAR;
	/// let f = p8::from<f32>(7.0_f32);
	///
	/// assert!(nar.is_nar());
	/// assert!(!f.is_nar());
	/// ```
	#[inline]
	pub fn is_nar(self) -> bool { self == NAR }

	/// Returns `true` if this value is `0.0`.
	///
	/// ```
	/// use positron::p8;
	///
	/// let nar = p8::ZERO;
	/// let f = p8::from<f32>(7.0_f32);
	///
	/// assert!(nar.is_zero());
	/// assert!(!f.is_zero());
	/// ```
	#[inline]
	pub fn is_zero(self) -> bool { self == ZERO }

	/// Computes the absolute value of `self`. Returns `NAR` if the number is `NAR`.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = p8::from(3.5_f32);
	/// let y = p8::from(-3.5_f32);
	///
	/// let abs_difference_x = (x.abs() - x).abs();
	/// let abs_difference_y = (y.abs() - (-y)).abs();
	///
	/// assert!(abs_difference_x <= p8::EPSILON);
	/// assert!(abs_difference_y <= p8::EPSILON);
	///
	/// assert!(p8::NAR.abs().is_nar());
	/// ```
	#[inline]
	fn abs(self) -> Self { if self.is_nar() { NAR } else { p8(self.0 & 0b0111_1111) } }

	/// Returns the posit category of the number. If only one property is going to be tested, it is generally faster to
	/// use the specific predicate instead.
	///
	/// ```
	/// use positron::{PositCategory, p8};
	///
	/// let zero = p8::ZERO;
	/// let inf = p8::NAR;
	/// let num = p8::from<f32>(12.4_f32);
	///
	/// assert_eq!(zero.classify(), PositCategory::Zero);
	/// assert_eq!(inf.classify(), PositCategory::Nar);
	/// assert_eq!(num.classify(), PositCategory::Normal);
	/// ```
	pub fn classify(self) -> PositCategory {
		match self {
			ZERO => PositCategory::Zero,
			NAR => PositCategory::Nar,
			_ => PositCategory::Normal,
		}
	}

	/// Returns `true` if `self` has a positive sign, and `false` is `self` is `NaR` or `0.0`.
	///
	/// ```
	/// let f = p8::from<f32>(7.0_f32);
	/// let g = p8::from<f32>(-7.0_f32);
	///
	/// assert!(f.is_sign_positive());
	/// assert!(!g.is_sign_positive());
	/// ```
	#[inline]
	pub fn is_sign_positive(self) -> bool {
		if self.is_zero() || self.is_nar() { false } else { !self.is_sign_negative() }
	}

	/// Returns `true` if `self` has a negative sign, and `false` is `self` is `NaR` or `0.0`.
	///
	/// ```
	/// let f = p8::from<f32>(7.0_f32);
	/// let g = p8::from<f32>(-7.0_f32);
	///
	/// assert!(!f.is_sign_negative());
	/// assert!(g.is_sign_negative());
	/// ```
	#[inline]
	pub fn is_sign_negative(self) -> bool {
		if self.is_zero() || self.is_nar() { false } else { self.0 & 0b1000_0000 == 0b1000_0000 }
	}

	/// Takes the reciprocal (inverse) of a number, `1/x`.
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = p8::from<f32>(2.0_f32);
	/// let abs_difference = (x.recip() - (p8::from(1.0) / x)).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn recip(self) -> Self {
		if self.is_nar() {
			NAR
		} else if self.is_zero() {
			NAR
		} else {
			p8(0b0100_0000 / self.0)
		}
	}

	/// Converts radians to degrees.
	///
	/// ```
	/// use positron::p8::{self, consts};
	///
	/// let angle = consts::PI;
	///
	/// let abs_difference = (angle.to_degrees() - p8::from(180.0f32)).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn to_degrees(self) -> Self {
		// Use a constant for better precision.
		const PIS_IN_180: p8 = p8(0b0111_1111); // 64
		self * PIS_IN_180
	}

	/// Converts degrees to radians.
	///
	/// ```
	/// use positron::p8::{self, consts};
	///
	/// let angle = p8::from(180.0f32);
	///
	/// let abs_difference = (angle.to_radians() - consts::PI).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn to_radians(self) -> Self { self * (consts::PI / Self::from(180.0f32)) }

	/// Returns the maximum of the two numbers.
	///
	/// ```
	/// let x = p8::from(1.0f32);
	/// let y = p8::from(2.0f32);
	///
	/// assert_eq!(x.max(y), y);
	/// ```
	///
	/// If one of the arguments is NaR, then the other argument is returned.
	#[inline]
	pub fn max(self, other: Self) -> Self {
		if self == NAR || other == NAR || self == other || self < other {
			other
		} else {
			self
		}
	}

	/// Returns the minimum of the two numbers.
	///
	/// ```
	/// let x = p8::from(1.0f32);
	/// let y = p8::from(2.0f32);
	///
	/// assert_eq!(x.min(y), x);
	/// ```
	///
	/// If one of the arguments is NaR, then the other argument is returned.
	#[inline]
	pub fn min(self, other: Self) -> Self {
		if self == NAR || other == NAR || self == other || other < self {
			other
		} else {
			self
		}
	}

	/// Raw transmutation to `u8`.
	///
	/// Note that this function is distinct from `as` casting, which attempts to preserve the *numeric* value, and not
	/// the bitwise value.
	///
	/// # Examples
	///
	/// ```
	/// assert_ne!(p8::from(1f32).to_bits(), p8::from(1f32) as u8); // to_bits() is not casting!
	/// assert_eq!(p8::from(0.75f32).to_bits(), 0b0011_0000);
	///
	/// ```
	#[inline]
	pub fn to_bits(self) -> u8 { unsafe { mem::transmute(self) } }

	/// Raw transmutation from `u8`.
	///
	/// Note that this function is distinct from `as` casting, which attempts to preserve the *numeric* value, and not
	/// the bitwise value.
	///
	/// # Examples
	///
	/// ```
	/// let v = p8::from_bits(0b0011_0000);
	/// assert_eq!(v, 0.75f32);
	/// ```
	#[inline]
	pub fn from_bits(x: u8) -> Self { unsafe { mem::transmute(x) } }

	/// Converts an integer from big endian to the target's endianness.
	/// On big endian this is a no-op. On little endian the bytes are swapped.
	///
	/// # Examples
	///
	/// Basic usage:
	///
	/// ```
	/// let n = p8::from(0x1A);
	///
	/// if cfg!(target_endian = "big") {
	///     assert_eq!(p8::from_be(n), n)
	/// } else {
	///     assert_eq!(p8::from_be(n), n.swap_bytes())
	/// }
	/// ```
	#[inline]
	pub const fn from_be(x: p8) -> Self {
		#[cfg(target_endian = "big")]
		{
			x
		}
		#[cfg(not(target_endian = "big"))]
		{
			Self(x.0.swap_bytes())
		}
	}

	/// Converts an integer from little endian to the target's endianness.
	///
	/// On little endian this is a no-op. On big endian the bytes are swapped.
	///
	/// # Examples
	///
	/// Basic usage:
	///
	/// ```
	/// let n = p8::from(0x1A);
	///
	/// if cfg!(target_endian = "little") {
	///     assert_eq!(p8::from_le(n), n)
	/// } else {
	///     assert_eq!(p8::from_le(n), n.swap_bytes())
	/// }
	/// ```
	#[inline]
	pub const fn from_le(x: p8) -> Self {
		#[cfg(target_endian = "little")]
		{
			x
		}
		#[cfg(not(target_endian = "little"))]
		{
			Self(x.0.swap_bytes())
		}
	}

	/// Create a posit from its representation as a byte array in big endian.
	///
	/// # Examples
	///
	/// ```
	/// let value = p8::from_be_bytes([0x48]);
	/// assert_eq!(value, p8::from(0.75_f32));
	/// ```
	#[inline]
	pub fn from_be_bytes(bytes: [p8; mem::size_of::<Self>()]) -> Self { Self::from_be(Self::from_ne_bytes(bytes)) }

	/// Create a posit from its representation as a byte array in little endian.
	///
	/// # Examples
	///
	/// ```
	/// let value = p8::from_le_bytes([0x48]);
	/// assert_eq!(value, p8::from(0.75_f32));
	/// ```
	#[inline]
	pub fn from_le_bytes(bytes: [p8; mem::size_of::<Self>()]) -> Self { Self::from_le(Self::from_ne_bytes(bytes)) }

	/// Create a posit from its representation as a byte array in native endian.
	///
	/// As the target platform's native endianness is used, portable code
	/// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
	/// appropriate instead.
	///
	/// [`from_be_bytes`]: #method.from_be_bytes
	/// [`from_le_bytes`]: #method.from_le_bytes
	///
	/// # Examples
	///
	/// ```
	/// let value = p8::from_ne_bytes(if cfg!(target_endian = "big") {
	///     [0x48]
	/// } else {
	///     [0x48]
	/// });
	/// assert_eq!(value, p8::from(0.75_f32));
	/// ```
	#[inline]
	pub fn from_ne_bytes(bytes: [p8; mem::size_of::<Self>()]) -> Self { unsafe { mem::transmute(bytes) } }

	/// Converts `self` to big endian from the target's endianness.
	/// On big endian this is a no-op. On little endian the bytes are swapped.
	///
	/// # Examples
	///
	/// ```
	/// let n = p8::from(0x1A);
	/// if cfg!(target_endian = "big") {
	///     assert_eq!(n.to_be(), n)
	/// } else {
	///     assert_eq!(n.to_be(), n.swap_bytes())
	/// }
	/// ```
	#[inline]
	pub fn to_be(self) -> Self { Self(self.0.to_be()) }

	/// Converts `self` to little endian from the target's endianness.
	/// On little endian this is a no-op. On big endian the bytes are swapped.
	///
	/// # Examples
	///
	/// ```
	/// let n = p8::from(0x1A);
	/// if cfg!(target_endian = "little") {
	///     assert_eq!(n.to_le(), n)
	/// } else {
	///     assert_eq!(n.to_le(), n.swap_bytes())
	/// }
	/// ```
	#[inline]
	pub fn to_le(self) -> Self { p8(self.0.to_le()) }

	/// Return the memory representation of this posit as a byte array in native byte order.
	///
	/// As the target platform's native endianness is used, portable code should use [`to_be_bytes`] or
	/// [`to_le_bytes`], as appropriate, instead.
	///
	/// [`to_be_bytes`]: #method.to_be_bytes
	/// [`to_le_bytes`]: #method.to_le_bytes
	///
	/// # Examples
	///
	/// ```
	/// let bytes = p8::from(0.75_f32).to_ne_bytes();
	/// assert_eq!(
	///     bytes,
	///     if cfg!(target_endian = "big") {
	///         [0x48]
	///     } else {
	///         [0x48]
	///     }
	/// );
	/// ```
	#[inline]
	pub fn to_be_bytes(self) -> [Self; mem::size_of::<Self>()] { self.to_be().to_ne_bytes() }

	/// Return the memory representation of this posit as a byte array in native byte order.
	///
	/// As the target platform's native endianness is used, portable code should use [`to_be_bytes`] or
	/// [`to_le_bytes`], as appropriate, instead.
	///
	/// [`to_be_bytes`]: #method.to_be_bytes
	/// [`to_le_bytes`]: #method.to_le_bytes
	///
	/// # Examples
	///
	/// ```
	/// let bytes = p8::from(0.75_f32).to_ne_bytes();
	/// assert_eq!(
	///     bytes,
	///     if cfg!(target_endian = "big") {
	///         [0x48]
	///     } else {
	///         [0x48]
	///     }
	/// );
	/// ```
	#[inline]
	pub fn to_le_bytes(self) -> [Self; mem::size_of::<Self>()] { self.to_le().to_ne_bytes() }

	/// Return the memory representation of this posit as a byte array in native byte order.
	///
	/// As the target platform's native endianness is used, portable code should use [`to_be_bytes`] or
	/// [`to_le_bytes`], as appropriate, instead.
	///
	/// [`to_be_bytes`]: #method.to_be_bytes
	/// [`to_le_bytes`]: #method.to_le_bytes
	///
	/// # Examples
	///
	/// ```
	/// let bytes = p8::from(0.75_f32).to_ne_bytes();
	/// assert_eq!(
	///     bytes,
	///     if cfg!(target_endian = "big") {
	///         [0x48]
	///     } else {
	///         [0x48]
	///     }
	/// );
	/// ```
	#[inline]
	pub fn to_ne_bytes(self) -> [Self; mem::size_of::<Self>()] { self.to_ne_bytes() }

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Returns the largest integer less than or equal to a number.
	///
	/// # Examples
	///
	/// ```
	/// let f = 3.7_f32;
	/// let g = 3.0_f32;
	/// let h = -3.7_f32;
	///
	/// assert_eq!(f.floor(), 3.0);
	/// assert_eq!(g.floor(), 3.0);
	/// assert_eq!(h.floor(), -4.0);
	/// ```
	#[inline]
	pub fn floor(self) -> Self {
		unimplemented!();
	}

	/// Returns the smallest integer greater than or equal to a number.
	///
	/// # Examples
	///
	/// ```
	/// let f = 3.01_f32;
	/// let g = 4.0_f32;
	///
	/// assert_eq!(f.ceil(), 4.0);
	/// assert_eq!(g.ceil(), 4.0);
	/// ```
	#[inline]
	pub fn ceil(self) -> Self {
		unimplemented!();
	}

	/// Returns the integer part of a number.
	///
	/// # Examples
	///
	/// ```
	/// let f = 3.7_f32;
	/// let g = 3.0_f32;
	/// let h = -3.7_f32;
	///
	/// assert_eq!(f.trunc(), 3.0);
	/// assert_eq!(g.trunc(), 3.0);
	/// assert_eq!(h.trunc(), -3.0);
	/// ```
	#[inline]
	pub fn trunc(self) -> Self {
		unimplemented!();
	}

	/// Returns the fractional part of a number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = p8::from(3.5_f32);
	/// let y = p8::from(-3.5_f32);
	/// let abs_difference_x = (x.fract() - 0.5).abs();
	/// let abs_difference_y = (y.fract() - (-0.5)).abs();
	///
	/// assert!(abs_difference_x <= p8::EPSILON);
	/// assert!(abs_difference_y <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn fract(self) -> Self {
		unimplemented!();
	}

	/// Returns a number composed of the magnitude of `self` and the sign of `sign`.
	///
	/// Equal to `self` if the sign of `self` and `sign` are the same, otherwise equal to `-self`. If `self` is a
	/// `NAR`, then a `NAR` with the sign of `sign` is returned.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let f = p8::from(3.5_f32);
	///
	/// assert_eq!(f.copysign(0.42), 3.5_f32);
	/// assert_eq!(f.copysign(-0.42), -3.5_f32);
	/// assert_eq!((-f).copysign(0.42), 3.5_f32);
	/// assert_eq!((-f).copysign(-0.42), -3.5_f32);
	///
	/// assert!(f32::NAR.copysign(1.0).is_nar());
	/// ```
	#[inline]
	#[must_use]
	pub fn copysign(self, sign: Self) -> Self {
		unimplemented!();
	}

	/// Fused multiply-add. Computes `(self * a) + b` with only one rounding error, yielding a more accurate result
	/// than an unfused multiply-add.
	///
	/// Using `mul_add` can be more performant than an unfused multiply-add if the target architecture has a dedicated
	/// `fma` CPU instruction.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let m = 10.0_f32;
	/// let x = 4.0_f32;
	/// let b = 60.0_f32;
	///
	/// // 100.0
	/// let abs_difference = (m.mul_add(x, b) - ((m * x) + b)).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn mul_add(self, a: Self, b: Self) -> Self {
		unimplemented!();
	}

	/// Calculates Euclidean division, the matching method for `rem_euclid`.
	///
	/// This computes the integer `n` such that `self = n * rhs + self.rem_euclid(rhs)`.
	/// In other words, the result is `self / rhs` rounded to the integer `n` such that `self >= n * rhs`.
	///
	/// # Examples
	///
	/// ```
	/// let a: f32 = 7.0;
	/// let b = 4.0;
	/// assert_eq!(a.div_euclid(b), 1.0); // 7.0 > 4.0 * 1.0
	/// assert_eq!((-a).div_euclid(b), -2.0); // -7.0 >= 4.0 * -2.0
	/// assert_eq!(a.div_euclid(-b), -1.0); // 7.0 >= -4.0 * -1.0
	/// assert_eq!((-a).div_euclid(-b), 2.0); // -7.0 >= -4.0 * 2.0
	/// ```
	#[inline]
	pub fn div_euclid(self, rhs: Self) -> Self {
		/*
		let q = (self / rhs).trunc();
		if self % rhs < ZERO {
			return if rhs > ZERO { q - 1.0 } else { q + 1.0 }
		}
		q*/
		unimplemented!();
	}

	/// Calculates the least nonnegative remainder of `self (mod rhs)`.
	///
	/// In particular, the return value `r` satisfies `0.0 <= r < rhs.abs()` in
	/// most cases. However, due to a floating point round-off error it can
	/// result in `r == rhs.abs()`, violating the mathematical definition, if
	/// `self` is much smaller than `rhs.abs()` in magnitude and `self < 0.0`.
	/// This result is not an element of the function's codomain, but it is the
	/// closest posit in the real numbers and thus fulfills the
	/// property `self == self.div_euclid(rhs) * rhs + self.rem_euclid(rhs)`
	/// approximatively.
	///
	/// # Examples
	///
	/// ```
	/// let a: f32 = 7.0;
	/// let b = 4.0;
	/// assert_eq!(a.rem_euclid(b), 3.0);
	/// assert_eq!((-a).rem_euclid(b), 1.0);
	/// assert_eq!(a.rem_euclid(-b), 3.0);
	/// assert_eq!((-a).rem_euclid(-b), 1.0);
	/// // limitation due to round-off error
	/// assert!((-std::p8::EPSILON).rem_euclid(3.0) != 0.0);
	/// ```
	#[inline]
	pub fn rem_euclid(self, rhs: Self) -> Self {
		/*
		let r = self % rhs;
		if r < 0.0 {
			r + rhs.abs()
		} else {
			r
		}
		*/
		unimplemented!();
	}

	/// Raises a number to an integer power.
	///
	/// Using this function is generally faster than using `powf`
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 2.0_f32;
	/// let abs_difference = (x.powi(2) - (x * x)).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn powi(self, n: i32) -> Self {
		unimplemented!();
	}

	/// Raises a number to a posit power.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 2.0_f32;
	/// let abs_difference = (x.powf(2.0) - (x * x)).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn powp(self, n: Self) -> Self {
		unimplemented!();
	}

	/// Takes the square root of a number.
	/// Returns NaR if `self` is a negative number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let positive = 4.0_f32;
	/// let negative = -4.0_f32;
	///
	/// let abs_difference = (positive.sqrt() - 2.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// assert!(negative.sqrt().is_nar());
	/// ```
	#[inline]
	pub fn sqrt(self) -> Self {
		if self < ZERO {
			NAR
		} else {
			unimplemented!();
		}
	}

	#[inline]
	pub fn rsqrt(self) -> Self {
		if self < ZERO {
			NAR
		} else {
			unimplemented!();
		}
	}

	/// Returns `e^(self)`, (the exponential function).
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let one = 1.0f32;
	/// // e^1
	/// let e = one.exp();
	///
	/// // ln(e) - 1 == 0
	/// let abs_difference = (e.ln() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn exp(self) -> Self {
		unimplemented!();
	}

	/// Returns `e^(self) - 1` in a way that is accurate even if the
	/// number is close to zero.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 6.0f32;
	///
	/// // e^(ln(6)) - 1
	/// let abs_difference = (x.ln().exp_m1() - 5.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn exp_m1(self) -> Self { self.exp() - 1 }

	/// Returns `2^(self)`.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let f = 2.0f32;
	///
	/// // 2^2 - 4 == 0
	/// let abs_difference = (f.exp2() - 4.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn exp2(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn exp2_m1(self) -> Self { self.exp2() - 1 }

	#[inline]
	pub fn exp10(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn exp10_m1(self) -> Self { self.exp10() - 1 }

	/// Returns the natural logarithm of the number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let one = 1.0f32;
	/// // e^1
	/// let e = one.exp();
	///
	/// // ln(e) - 1 == 0
	/// let abs_difference = (e.ln() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn ln(self) -> Self {
		unimplemented!();
	}

	/// Returns `ln(1+n)` (natural logarithm) more accurately than if
	/// the operations were performed separately.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = f32::consts::E - 1.0;
	///
	/// // ln(1 + (e - 1)) == ln(e) == 1
	/// let abs_difference = (x.ln_1p() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn ln_1p(self) -> Self {
		unimplemented!();
	}

	/// Returns the logarithm of the number with respect to an arbitrary base.
	///
	/// The result may not be correctly rounded owing to implementation details;
	/// `self.log2()` can produce more accurate results for base 2, and
	/// `self.log10()` can produce more accurate results for base 10.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let five = 5.0f32;
	///
	/// // log5(5) - 1 == 0
	/// let abs_difference = (five.log(5.0) - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn log(self, base: Self) -> Self { if self <= 0 { NAR } else { self.ln() / base.ln() } }

	#[inline]
	pub fn log_p1(self, base: Self) -> Self { if self <= -1 { NAR } else { log(self + 1, base) } }

	/// Returns the base 2 logarithm of the number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let two = 2.0f32;
	///
	/// // log2(2) - 1 == 0
	/// let abs_difference = (two.log2() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn log2(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn log2_p1(self) -> Self { (self + 1).log2() }

	/// Returns the base 10 logarithm of the number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let ten = 10.0f32;
	///
	/// // log10(10) - 1 == 0
	/// let abs_difference = (ten.log10() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn log10(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn log10_p1(self) -> Self {(self + 1).log10() }

	/// Takes the cubic root of a number.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 8.0f32;
	///
	/// // x^(1/3) - 2 == 0
	/// let abs_difference = (x.cbrt() - 2.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn cbrt(self) -> Self {
		unimplemented!();
	}

	/// Calculates the length of the hypotenuse of a right-angle triangle given
	/// legs of length `x` and `y`.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 2.0f32;
	/// let y = 3.0f32;
	///
	/// // sqrt(x^2 + y^2)
	/// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn hypot(self, other: Self) -> Self { (self.powi(2) + other.powi(2)).sqrt() }

	/// Computes the sine of a number (in radians).
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = f32::consts::FRAC_PI_2;
	///
	/// let abs_difference = (x.sin() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn sin(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn sinpi(self) -> Self { (self * PI).sin() }

	/// Computes the cosine of a number (in radians).
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 2.0 * f32::consts::PI;
	///
	/// let abs_difference = (x.cos() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn cos(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn cospi(self) -> Self { (self * PI).cos() }

	/// Computes the tangent of a number (in radians).
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = f32::consts::FRAC_PI_4;
	/// let abs_difference = (x.tan() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn tan(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn tanpi(self) -> Self { (self * PI).tan() }

	/// Computes the arcsine of a number. Return value is in radians in
	/// the range [-pi/2, pi/2] or NaN if the number is outside the range
	/// [-1, 1].
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let f = f32::consts::FRAC_PI_2;
	///
	/// // asin(sin(pi/2))
	/// let abs_difference = (f.sin().asin() - f32::consts::FRAC_PI_2).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn asin(self) -> Self {
		if 1 < self.abs() {
			NAR
		} else {
			unimplemented!();
		}
	}

	#[inline]
	pub fn asinpi(self) -> Self { if 1 < self.abs() { NAR } else { (self / PI).asin() } }

	/// Computes the arccosine of a number. Return value is in radians in
	/// the range [0, pi] or NaN if the number is outside the range
	/// [-1, 1].
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let f = f32::consts::FRAC_PI_4;
	///
	/// // acos(cos(pi/4))
	/// let abs_difference = (f.cos().acos() - f32::consts::FRAC_PI_4).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn acos(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn acospi(self) -> Self { if 1 < self.abs() { NAR } else { (self / PI).acos() } }

	/// Computes the arctangent of a number. Return value is in radians in the
	/// range [-pi/2, pi/2];
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let f = 1.0f32;
	///
	/// // atan(tan(1))
	/// let abs_difference = (f.tan().atan() - 1.0).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn atan(self) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn atanpi(self) -> Self { (self / PI).atan() }

	/// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
	///
	/// * `x = 0`, `y = 0`: `0`
	/// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
	/// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
	/// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// // Positive angles measured counter-clockwise
	/// // from positive x axis
	/// // -pi/4 radians (45 deg clockwise)
	/// let x1 = 3.0f32;
	/// let y1 = -3.0f32;
	///
	/// // 3pi/4 radians (135 deg counter-clockwise)
	/// let x2 = -3.0f32;
	/// let y2 = 3.0f32;
	///
	/// let abs_difference_1 = (y1.atan2(x1) - (-f32::consts::FRAC_PI_4)).abs();
	/// let abs_difference_2 = (y2.atan2(x2) - (3.0 * f32::consts::FRAC_PI_4)).abs();
	///
	/// assert!(abs_difference_1 <= p8::EPSILON);
	/// assert!(abs_difference_2 <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn atan2(self, other: Self) -> Self {
		unimplemented!();
	}

	/// Simultaneously computes the sine and cosine of the number, `x`. Returns
	/// `(sin(x), cos(x))`.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = f32::consts::FRAC_PI_4;
	/// let f = x.sin_cos();
	///
	/// let abs_difference_0 = (f.0 - x.sin()).abs();
	/// let abs_difference_1 = (f.1 - x.cos()).abs();
	///
	/// assert!(abs_difference_0 <= p8::EPSILON);
	/// assert!(abs_difference_1 <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn sin_cos(self) -> (Self, Self) {
		(self.sin(), self.cos())
	}

	/// Hyperbolic sine function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let e = f32::consts::E;
	/// let x = 1.0f32;
	///
	/// let f = x.sinh();
	/// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
	/// let g = ((e * e) - 1.0) / (2.0 * e);
	/// let abs_difference = (f - g).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn sinh(self) -> Self {
		unimplemented!();
	}

	/// Hyperbolic cosine function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let e = f32::consts::E;
	/// let x = 1.0f32;
	/// let f = x.cosh();
	/// // Solving cosh() at 1 gives this result
	/// let g = ((e * e) + 1.0) / (2.0 * e);
	/// let abs_difference = (f - g).abs();
	///
	/// // Same result
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn cosh(self) -> Self {
		unimplemented!();
	}

	/// Hyperbolic tangent function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let e = f32::consts::E;
	/// let x = 1.0f32;
	///
	/// let f = x.tanh();
	/// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
	/// let g = (1.0 - e.powi(-2)) / (1.0 + e.powi(-2));
	/// let abs_difference = (f - g).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn tanh(self) -> Self {
		unimplemented!();
	}

	/// Inverse hyperbolic sine function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 1.0f32;
	/// let f = x.sinh().asinh();
	///
	/// let abs_difference = (f - x).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn asinh(self) -> Self { if 1 < self.abs() { NAR } else { self.asin() } }

	/// Inverse hyperbolic cosine function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let x = 1.0f32;
	/// let f = x.cosh().acosh();
	///
	/// let abs_difference = (f - x).abs();
	///
	/// assert!(abs_difference <= p8::EPSILON);
	/// ```
	#[inline]
	pub fn acosh(self) -> Self { if 1 < self.abs() { NAR } else { self.asin() } }

	/// Inverse hyperbolic tangent function.
	///
	/// # Examples
	///
	/// ```
	/// use positron::p8;
	///
	/// let e = f32::consts::E;
	/// let f = e.tanh().atanh();
	///
	/// let abs_difference = (f - e).abs();
	///
	/// assert!(abs_difference <= 1e-5);
	/// ```
	#[inline]
	pub fn atanh(self) -> Self {
		//0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
		unimplemented!();
	}

	/// Restrict a value to a certain interval unless it is NaN.
	///
	/// Returns `max` if `self` is greater than `max`, and `min` if `self` is less than `min`. Otherwise this returns
	/// `self`.
	///
	/// Not that this function returns NaN if the initial value was NaN as well.
	///
	/// # Panics
	///
	/// Panics if `min > max`, `min` is NaN, or `max` is NaN.
	///
	/// # Examples
	///
	/// ```
	/// #![feature(clamp)]
	/// assert!((-3.0f32).clamp(-2.0, 1.0) == -2.0);
	/// assert!((0.0f32).clamp(-2.0, 1.0) == 0.0);
	/// assert!((2.0f32).clamp(-2.0, 1.0) == 1.0);
	/// assert!((std::f32::NAR).clamp(-2.0, 1.0).is_nar());
	/// ```
	#[inline]
	pub fn clamp(self, min: Self, max: Self) -> Self {
		assert!(min <= max);
		let mut x = self;
		if x < min { x = min; }
		if x > max { x = max; }
		x
	}

	#[inline]
	pub fn compound(self, i: u32) -> Self {
		unimplemented!();
	}

	#[inline]
	pub fn rooti(self, i: u32) -> Self {
		unimplemented!();
	}
}

pub trait Round<T> {
	fn round(x: T) -> Self;
}

impl Round<Self> for p8 {
	/// Converts posit to the nearest posit with integer value, and the nearest even integer if two integers are
	/// equally far from posit
	fn round(x: Self) -> Self {
		unimplemented!();
	}
}

impl Round<f32> for p8 {
	/*
	For all other values, the value is rounded to the nearest binary value if the posit were encoded to
	infinite precision beyond the nbits length; if two posits are equally near, the one with binary encoding
	ending in 0 is selected.
	Note:  Rule (4) has the effect of rounding to the posit with the nearest logarithm when the dropped bitis
	an exponent bit, and to the nearest posit by absolute difference in other cases.
	*/
	fn round(x: f32) -> Self {
		unimplemented!();
		/*
		if MAXPOS < float.abs() {
			// TODO: sign(x) * maxpos
			p8(0b1111_1111)
		} else if 0 < float.abs() && float.abs() < MINPOS {
			p8(self.sign() * MINPOS)
		} else {
			ZERO
		}
		*/
	}
}

impl Round<f64> for p8 {
	fn round(x: f64) -> Self {
		unimplemented!();
	}
}

impl From<f32> for p8 {
	fn from(x: f32) -> Self {
		unimplemented!();
	}
}

impl From<f64> for p8 {
	fn from(float: f64) -> Self {
		unimplemented!();
	}
}

impl From<i8> for p8 {
	fn from(x: i8) -> Self {
		Self(x)
	}
}

impl From<u8> for p8 {
	fn from(x: u8) -> Self {
		unimplemented!();
	}
}

impl Posit for p8 {}
/*
impl Add for p8 {
	type Output = Self;

	fn add(self, other: p8) -> Self::Output {
		unimplemented!();
	}
}

impl AddAssign for p8 {
	fn add_assign(&mut self, other: Self) {
		// *self = p8(self.0 + other.0);
		unimplemented!();
	}
}

impl Div for p8 {
	// The division of rational numbers is a closed operation.
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output {
		if rhs == ZERO {
			NAR
		}
		unimplemented!();
	}
}

impl DivAssign<p8> for p8 {
	fn div_assign(&mut self, rhs: Self) {
		if rhs == ZERO {
			NAR
		}
		//self.0 /= rhs.0;
		unimplemented!();
	}
}

impl Index<usize> for p8 {
	type Output = i8;

	fn index(&self, index: usize) -> &Self::Output {
		match index {
			0 => &(&self.0 & 0b1000_0000),
			1 => &(&self.0 & 0b0100_0000),
			2 => &(&self.0 & 0b0010_0000),
			3 => &(&self.0 & 0b0001_0000),
			4 => &(&self.0 & 0b0000_1000),
			5 => &(&self.0 & 0b0000_0100),
			6 => &(&self.0 & 0b0000_0010),
			7 => &(&self.0 & 0b0000_0001),
			_ => panic!(),
		}
	}
}

impl IndexMut<usize> for p8 {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		match index {
			0 => &mut (&self.0 & 0b1000_0000),
			1 => &mut (&self.0 & 0b0100_0000),
			2 => &mut (&self.0 & 0b0010_0000),
			3 => &mut (&self.0 & 0b0001_0000),
			4 => &mut (&self.0 & 0b0000_1000),
			5 => &mut (&self.0 & 0b0000_0100),
			6 => &mut (&self.0 & 0b0000_0010),
			7 => &mut (&self.0 & 0b0000_0001),
			_ => panic!(),
		}
	}
}

impl Mul for p8 {
	// The multiplication of rational numbers is a closed operation.
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		unimplemented!();
	}
}

impl MulAssign<p8> for p8 {
	fn mul_assign(&mut self, rhs: Self) {
		//self.0 *= rhs.0;
		unimplemented!();
	}
}

impl Neg for p8 {
	type Output = Self;

	fn neg(self) -> Self::Output {
		//p8(self.0.neg())
		unimplemented!();
	}
}

impl Rem<p8> for p8 {
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output {
		//p8(self.0 % rhs.0)
		unimplemented!();
	}
}

impl RemAssign<p8> for p8 {
	fn rem_assign(&mut self, rhs: Self) {
		//self.0 %= rhs.0;
		unimplemented!();
	}
}

impl Sub for p8 {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		//p8(self.0 - rhs.0)
		unimplemented!();
	}
}

impl SubAssign for p8 {
	fn sub_assign(&mut self, rhs: Self) {
		//self.0 = self.0 - rhs.0;
		unimplemented!();
	}
}
unsafe impl Send for p8 {}
unsafe impl Sync for p8 {}
*/

#[cfg(test)]
mod tests {
	#[test]
	fn it_works() {
		assert_eq!(2 + 2, 4);
	}
}