var searchIndex = {};
searchIndex["range_map"] = {"doc":"","items":[[3,"Range","range_map","A range of elements, including the endpoints.",null,null],[12,"start","","",0,null],[12,"end","","",0,null],[3,"RangeMap","","A set of characters. Optionally, each character in the set may be associated with some data.",null,null],[3,"PairIter","","",null,null],[3,"RangeSet","","A set of integers, implemented as a sorted list of (inclusive) ranges.",null,null],[3,"RangeIter","","",null,null],[3,"EltIter","","",null,null],[3,"RangeMultiMap","","A multi-valued mapping from primitive integers to other data.",null,null],[11,"cmp","","",0,null],[11,"partial_cmp","","",0,null],[11,"lt","","",0,null],[11,"le","","",0,null],[11,"gt","","",0,null],[11,"ge","","",0,null],[11,"eq","","",0,null],[11,"ne","","",0,null],[11,"hash","","",0,null],[11,"clone","","",0,null],[11,"fmt","","",0,null],[11,"new","","Creates a new range with the given start and endpoints (inclusive).",0,{"inputs":[{"name":"t"},{"name":"t"}],"output":{"name":"range"}}],[11,"full","","Creates a new range containing everything.",0,{"inputs":[],"output":{"name":"range"}}],[11,"single","","Creates a new range containing a single thing.",0,{"inputs":[{"name":"t"}],"output":{"name":"range"}}],[11,"contains","","Tests whether a given element belongs to this range.",0,null],[11,"intersects","","Checks whether the intersections overlap.",0,null],[11,"intersection","","Computes the intersection between two ranges. Returns none if the intersection is empty.",0,null],[11,"cover","","Returns the smallest range that covers `self` and `other`.",0,null],[11,"eq","","",0,null],[11,"partial_cmp","","",0,null],[11,"eq","","",1,null],[11,"ne","","",1,null],[11,"hash","","",1,null],[11,"clone","","",1,null],[11,"fmt","","",1,null],[11,"from_iter","","Builds a `RangeMap` from an iterator over pairs. If any ranges overlap, they must map to\nthe same value.",1,{"inputs":[{"name":"i"}],"output":{"name":"self"}}],[11,"new","","Creates a new empty `RangeMap`.",1,{"inputs":[],"output":{"name":"rangemap"}}],[11,"from_sorted_vec","","Creates a `RangeMap` from a `Vec`, which must contain ranges in ascending order. If any\nranges overlap, they must map to the same value.",1,{"inputs":[{"name":"vec"}],"output":{"name":"rangemap"}}],[11,"num_ranges","","Returns the number of mapped ranges.",1,null],[11,"is_empty","","Tests whether this map is empty.",1,null],[11,"is_full","","Tests whether this `CharMap` maps every value.",1,null],[11,"ranges_values","","Iterates over all the mapped ranges and values.",1,null],[11,"keys_values","","Iterates over all mappings.",1,null],[11,"get","","Finds the value that `x` maps to, if it exists.",1,null],[11,"intersection","","Returns those mappings whose keys belong to the given set.",1,null],[11,"num_keys","","Counts the number of mapped keys.",1,null],[11,"to_range_set","","Returns the set of mapped chars, forgetting what they are mapped to.",1,null],[11,"map_values","","Modifies the values in place.",1,null],[11,"retain_values","","Modifies this map to contain only those mappings with values `v` satisfying `f(v)`.",1,null],[11,"as_mut_slice","","Returns a mutable view into this map.",1,null],[11,"fmt","","",2,null],[11,"clone","","",2,null],[11,"next","","",2,null],[11,"eq","","",3,null],[11,"ne","","",3,null],[11,"hash","","",3,null],[11,"clone","","",3,null],[11,"eq","","",4,null],[11,"ne","","",4,null],[11,"hash","","",4,null],[11,"fmt","","",4,null],[11,"clone","","",4,null],[11,"next","","",4,null],[11,"fmt","","",5,null],[11,"clone","","",5,null],[11,"next","","",5,null],[11,"fmt","","",3,null],[11,"from_iter","","Builds a `RangeSet` from an iterator over `Range`s.",3,{"inputs":[{"name":"i"}],"output":{"name":"self"}}],[11,"new","","Creates a new empty `RangeSet`.",3,{"inputs":[],"output":{"name":"rangeset"}}],[11,"is_empty","","Tests if this set is empty.",3,null],[11,"is_full","","Tests whether this set contains every valid value of `T`.",3,null],[11,"num_ranges","","Returns the number of ranges used to represent this set.",3,null],[11,"num_elements","","Returns the number of elements in the set.",3,null],[11,"ranges","","Returns an iterator over all ranges in this set.",3,null],[11,"elements","","Returns an iterator over all elements in this set.",3,null],[11,"contains","","Checks if this set contains a value.",3,null],[11,"union","","Returns the union between `self` and `other`.",3,null],[11,"full","","Creates a set that contains every value of `T`.",3,{"inputs":[],"output":{"name":"rangeset"}}],[11,"single","","Creates a set containing a single element.",3,{"inputs":[{"name":"t"}],"output":{"name":"rangeset"}}],[11,"except","","Creates a set containing all elements except the given ones.",3,{"inputs":[{"name":"i"}],"output":{"name":"rangeset"}}],[11,"intersection","","Finds the intersection between this set and `other`.",3,null],[11,"negated","","Returns the set of all characters that are not in this set.",3,null],[11,"eq","","",6,null],[11,"ne","","",6,null],[11,"hash","","",6,null],[11,"clone","","",6,null],[11,"from_iter","","Builds a `RangeMultiMap` from an iterator over `Range` and values..",6,{"inputs":[{"name":"i"}],"output":{"name":"self"}}],[11,"fmt","","",6,null],[11,"new","","Creates a new empty map.",6,{"inputs":[],"output":{"name":"rangemultimap"}}],[11,"num_ranges","","Returns the number of mapped ranges.",6,null],[11,"is_empty","","Checks if the map is empty.",6,null],[11,"insert","","Adds a new mapping from a range of characters to `value`.",6,null],[11,"from_vec","","Creates a map from a vector of pairs.",6,{"inputs":[{"name":"vec"}],"output":{"name":"rangemultimap"}}],[11,"intersection","","Returns a new `RangeMultiMap` containing only the mappings for keys that belong to the\ngiven set.",6,null],[11,"map_values","","",6,null],[11,"retain_values","","Modifies this map in place to only contain mappings whose values `v` satisfy `f(v)`.",6,null],[11,"into_vec","","Returns the underlying `Vec`.",6,null],[11,"ranges_values","","Iterates over all the mapped ranges and values.",6,null],[11,"group","","Makes the ranges sorted and non-overlapping. The data associated with each range will\nbe a `Vec&lt;T&gt;` instead of a single `T`.",6,null]],"paths":[[3,"Range"],[3,"RangeMap"],[3,"PairIter"],[3,"RangeSet"],[3,"RangeIter"],[3,"EltIter"],[3,"RangeMultiMap"]]};
searchIndex["num_traits"] = {"doc":"Numeric traits for generic mathematics","items":[[3,"ParseFloatError","num_traits","",null,null],[12,"kind","","",0,null],[4,"FloatErrorKind","","",null,null],[13,"Empty","","",1,null],[13,"Invalid","","",1,null],[0,"identities","","",null,null],[5,"zero","num_traits::identities","Returns the additive identity, `0`.",null,{"inputs":[],"output":{"name":"t"}}],[5,"one","","Returns the multiplicative identity, `1`.",null,{"inputs":[],"output":{"name":"t"}}],[8,"Zero","","Defines an additive identity element for `Self`.",null,null],[10,"zero","","Returns the additive identity element of `Self`, `0`.",2,{"inputs":[],"output":{"name":"self"}}],[10,"is_zero","","Returns `true` if `self` is equal to the additive identity.",2,null],[8,"One","","Defines a multiplicative identity element for `Self`.",null,null],[10,"one","","Returns the multiplicative identity element of `Self`, `1`.",3,{"inputs":[],"output":{"name":"self"}}],[0,"sign","num_traits","",null,null],[5,"abs","num_traits::sign","Computes the absolute value.",null,{"inputs":[{"name":"t"}],"output":{"name":"t"}}],[5,"abs_sub","","The positive difference of two numbers.",null,{"inputs":[{"name":"t"},{"name":"t"}],"output":{"name":"t"}}],[5,"signum","","Returns the sign of the number.",null,{"inputs":[{"name":"t"}],"output":{"name":"t"}}],[8,"Signed","","Useful functions for signed numbers (i.e. numbers that can be negative).",null,null],[10,"abs","","Computes the absolute value.",4,null],[10,"abs_sub","","The positive difference of two numbers.",4,null],[10,"signum","","Returns the sign of the number.",4,null],[10,"is_positive","","Returns true if the number is positive and false if the number is zero or negative.",4,null],[10,"is_negative","","Returns true if the number is negative and false if the number is zero or positive.",4,null],[8,"Unsigned","","A trait for values which cannot be negative",null,null],[0,"ops","num_traits","",null,null],[0,"saturating","num_traits::ops","",null,null],[8,"Saturating","num_traits::ops::saturating","Saturating math operations",null,null],[10,"saturating_add","","Saturating addition operator.\nReturns a+b, saturating at the numeric bounds instead of overflowing.",5,null],[10,"saturating_sub","","Saturating subtraction operator.\nReturns a-b, saturating at the numeric bounds instead of overflowing.",5,null],[0,"checked","num_traits::ops","",null,null],[8,"CheckedAdd","num_traits::ops::checked","Performs addition that returns `None` instead of wrapping around on\noverflow.",null,null],[10,"checked_add","","Adds two numbers, checking for overflow. If overflow happens, `None` is\nreturned.",6,null],[8,"CheckedSub","","Performs subtraction that returns `None` instead of wrapping around on underflow.",null,null],[10,"checked_sub","","Subtracts two numbers, checking for underflow. If underflow happens,\n`None` is returned.",7,null],[8,"CheckedMul","","Performs multiplication that returns `None` instead of wrapping around on underflow or\noverflow.",null,null],[10,"checked_mul","","Multiplies two numbers, checking for underflow or overflow. If underflow\nor overflow happens, `None` is returned.",8,null],[8,"CheckedDiv","","Performs division that returns `None` instead of panicking on division by zero and instead of\nwrapping around on underflow and overflow.",null,null],[10,"checked_div","","Divides two numbers, checking for underflow, overflow and division by\nzero. If any of that happens, `None` is returned.",9,null],[0,"bounds","num_traits","",null,null],[8,"Bounded","num_traits::bounds","Numbers which have upper and lower bounds",null,null],[10,"min_value","","returns the smallest finite number this type can represent",10,{"inputs":[],"output":{"name":"self"}}],[10,"max_value","","returns the largest finite number this type can represent",10,{"inputs":[],"output":{"name":"self"}}],[0,"float","num_traits","",null,null],[8,"Float","num_traits::float","",null,null],[10,"nan","","Returns the `NaN` value.",11,{"inputs":[],"output":{"name":"self"}}],[10,"infinity","","Returns the infinite value.",11,{"inputs":[],"output":{"name":"self"}}],[10,"neg_infinity","","Returns the negative infinite value.",11,{"inputs":[],"output":{"name":"self"}}],[10,"neg_zero","","Returns `-0.0`.",11,{"inputs":[],"output":{"name":"self"}}],[10,"min_value","","Returns the smallest finite value that this type can represent.",11,{"inputs":[],"output":{"name":"self"}}],[10,"min_positive_value","","Returns the smallest positive, normalized value that this type can represent.",11,{"inputs":[],"output":{"name":"self"}}],[10,"max_value","","Returns the largest finite value that this type can represent.",11,{"inputs":[],"output":{"name":"self"}}],[10,"is_nan","","Returns `true` if this value is `NaN` and false otherwise.",11,null],[10,"is_infinite","","Returns `true` if this value is positive infinity or negative infinity and\nfalse otherwise.",11,null],[10,"is_finite","","Returns `true` if this number is neither infinite nor `NaN`.",11,null],[10,"is_normal","","Returns `true` if the number is neither zero, infinite,\n[subnormal][subnormal], or `NaN`.",11,null],[10,"classify","","Returns the floating point category of the number. If only one property\nis going to be tested, it is generally faster to use the specific\npredicate instead.",11,null],[10,"floor","","Returns the largest integer less than or equal to a number.",11,null],[10,"ceil","","Returns the smallest integer greater than or equal to a number.",11,null],[10,"round","","Returns the nearest integer to a number. Round half-way cases away from\n`0.0`.",11,null],[10,"trunc","","Return the integer part of a number.",11,null],[10,"fract","","Returns the fractional part of a number.",11,null],[10,"abs","","Computes the absolute value of `self`. Returns `Float::nan()` if the\nnumber is `Float::nan()`.",11,null],[10,"signum","","Returns a number that represents the sign of `self`.",11,null],[10,"is_sign_positive","","Returns `true` if `self` is positive, including `+0.0` and\n`Float::infinity()`.",11,null],[10,"is_sign_negative","","Returns `true` if `self` is negative, including `-0.0` and\n`Float::neg_infinity()`.",11,null],[10,"mul_add","","Fused multiply-add. Computes `(self * a) + b` with only one rounding\nerror. This produces a more accurate result with better performance than\na separate multiplication operation followed by an add.",11,null],[10,"recip","","Take the reciprocal (inverse) of a number, `1/x`.",11,null],[10,"powi","","Raise a number to an integer power.",11,null],[10,"powf","","Raise a number to a floating point power.",11,null],[10,"sqrt","","Take the square root of a number.",11,null],[10,"exp","","Returns `e^(self)`, (the exponential function).",11,null],[10,"exp2","","Returns `2^(self)`.",11,null],[10,"ln","","Returns the natural logarithm of the number.",11,null],[10,"log","","Returns the logarithm of the number with respect to an arbitrary base.",11,null],[10,"log2","","Returns the base 2 logarithm of the number.",11,null],[10,"log10","","Returns the base 10 logarithm of the number.",11,null],[11,"to_degrees","","Converts radians to degrees.",11,null],[11,"to_radians","","Converts degrees to radians.",11,null],[10,"max","","Returns the maximum of the two numbers.",11,null],[10,"min","","Returns the minimum of the two numbers.",11,null],[10,"abs_sub","","The positive difference of two numbers.",11,null],[10,"cbrt","","Take the cubic root of a number.",11,null],[10,"hypot","","Calculate the length of the hypotenuse of a right-angle triangle given\nlegs of length `x` and `y`.",11,null],[10,"sin","","Computes the sine of a number (in radians).",11,null],[10,"cos","","Computes the cosine of a number (in radians).",11,null],[10,"tan","","Computes the tangent of a number (in radians).",11,null],[10,"asin","","Computes the arcsine of a number. Return value is in radians in\nthe range [-pi/2, pi/2] or NaN if the number is outside the range\n[-1, 1].",11,null],[10,"acos","","Computes the arccosine of a number. Return value is in radians in\nthe range [0, pi] or NaN if the number is outside the range\n[-1, 1].",11,null],[10,"atan","","Computes the arctangent of a number. Return value is in radians in the\nrange [-pi/2, pi/2];",11,null],[10,"atan2","","Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`).",11,null],[10,"sin_cos","","Simultaneously computes the sine and cosine of the number, `x`. Returns\n`(sin(x), cos(x))`.",11,null],[10,"exp_m1","","Returns `e^(self) - 1` in a way that is accurate even if the\nnumber is close to zero.",11,null],[10,"ln_1p","","Returns `ln(1+n)` (natural logarithm) more accurately than if\nthe operations were performed separately.",11,null],[10,"sinh","","Hyperbolic sine function.",11,null],[10,"cosh","","Hyperbolic cosine function.",11,null],[10,"tanh","","Hyperbolic tangent function.",11,null],[10,"asinh","","Inverse hyperbolic sine function.",11,null],[10,"acosh","","Inverse hyperbolic cosine function.",11,null],[10,"atanh","","Inverse hyperbolic tangent function.",11,null],[10,"integer_decode","","Returns the mantissa, base 2 exponent, and sign as integers, respectively.\nThe original number can be recovered by `sign * mantissa * 2 ^ exponent`.\nThe floating point encoding is documented in the [Reference][floating-point].",11,null],[0,"cast","num_traits","",null,null],[5,"cast","num_traits::cast","Cast from one machine scalar to another.",null,{"inputs":[{"name":"t"}],"output":{"name":"option"}}],[8,"ToPrimitive","","A generic trait for converting a value to a number.",null,null],[11,"to_isize","","Converts the value of `self` to an `isize`.",12,null],[11,"to_i8","","Converts the value of `self` to an `i8`.",12,null],[11,"to_i16","","Converts the value of `self` to an `i16`.",12,null],[11,"to_i32","","Converts the value of `self` to an `i32`.",12,null],[10,"to_i64","","Converts the value of `self` to an `i64`.",12,null],[11,"to_usize","","Converts the value of `self` to a `usize`.",12,null],[11,"to_u8","","Converts the value of `self` to an `u8`.",12,null],[11,"to_u16","","Converts the value of `self` to an `u16`.",12,null],[11,"to_u32","","Converts the value of `self` to an `u32`.",12,null],[10,"to_u64","","Converts the value of `self` to an `u64`.",12,null],[11,"to_f32","","Converts the value of `self` to an `f32`.",12,null],[11,"to_f64","","Converts the value of `self` to an `f64`.",12,null],[8,"FromPrimitive","","A generic trait for converting a number to a value.",null,null],[11,"from_isize","","Convert an `isize` to return an optional value of this type. If the\nvalue cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"isize"}],"output":{"name":"option"}}],[11,"from_i8","","Convert an `i8` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"i8"}],"output":{"name":"option"}}],[11,"from_i16","","Convert an `i16` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"i16"}],"output":{"name":"option"}}],[11,"from_i32","","Convert an `i32` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"i32"}],"output":{"name":"option"}}],[10,"from_i64","","Convert an `i64` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"i64"}],"output":{"name":"option"}}],[11,"from_usize","","Convert a `usize` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"usize"}],"output":{"name":"option"}}],[11,"from_u8","","Convert an `u8` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"u8"}],"output":{"name":"option"}}],[11,"from_u16","","Convert an `u16` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"u16"}],"output":{"name":"option"}}],[11,"from_u32","","Convert an `u32` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"u32"}],"output":{"name":"option"}}],[10,"from_u64","","Convert an `u64` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"u64"}],"output":{"name":"option"}}],[11,"from_f32","","Convert a `f32` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"f32"}],"output":{"name":"option"}}],[11,"from_f64","","Convert a `f64` to return an optional value of this type. If the\ntype cannot be represented by this value, the `None` is returned.",13,{"inputs":[{"name":"f64"}],"output":{"name":"option"}}],[8,"NumCast","","An interface for casting between machine scalars.",null,null],[10,"from","","Creates a number from another value that can be converted into\na primitive via the `ToPrimitive` trait.",14,{"inputs":[{"name":"t"}],"output":{"name":"option"}}],[0,"int","num_traits","",null,null],[8,"PrimInt","num_traits::int","",null,null],[10,"count_ones","","Returns the number of ones in the binary representation of `self`.",15,null],[10,"count_zeros","","Returns the number of zeros in the binary representation of `self`.",15,null],[10,"leading_zeros","","Returns the number of leading zeros in the binary representation\nof `self`.",15,null],[10,"trailing_zeros","","Returns the number of trailing zeros in the binary representation\nof `self`.",15,null],[10,"rotate_left","","Shifts the bits to the left by a specified amount amount, `n`, wrapping\nthe truncated bits to the end of the resulting integer.",15,null],[10,"rotate_right","","Shifts the bits to the right by a specified amount amount, `n`, wrapping\nthe truncated bits to the beginning of the resulting integer.",15,null],[10,"signed_shl","","Shifts the bits to the left by a specified amount amount, `n`, filling\nzeros in the least significant bits.",15,null],[10,"signed_shr","","Shifts the bits to the right by a specified amount amount, `n`, copying\nthe &quot;sign bit&quot; in the most significant bits even for unsigned types.",15,null],[10,"unsigned_shl","","Shifts the bits to the left by a specified amount amount, `n`, filling\nzeros in the least significant bits.",15,null],[10,"unsigned_shr","","Shifts the bits to the right by a specified amount amount, `n`, filling\nzeros in the most significant bits.",15,null],[10,"swap_bytes","","Reverses the byte order of the integer.",15,null],[10,"from_be","","Convert an integer from big endian to the target&#39;s endianness.",15,{"inputs":[{"name":"self"}],"output":{"name":"self"}}],[10,"from_le","","Convert an integer from little endian to the target&#39;s endianness.",15,{"inputs":[{"name":"self"}],"output":{"name":"self"}}],[10,"to_be","","Convert `self` to big endian from the target&#39;s endianness.",15,null],[10,"to_le","","Convert `self` to little endian from the target&#39;s endianness.",15,null],[10,"pow","","Raises self to the power of `exp`, using exponentiation by squaring.",15,null],[0,"pow","num_traits","",null,null],[5,"pow","num_traits::pow","Raises a value to the power of exp, using exponentiation by squaring.",null,{"inputs":[{"name":"t"},{"name":"usize"}],"output":{"name":"t"}}],[5,"checked_pow","","Raises a value to the power of exp, returning `None` if an overflow occurred.",null,{"inputs":[{"name":"t"},{"name":"usize"}],"output":{"name":"option"}}],[8,"Num","num_traits","The base trait for numeric types",null,null],[16,"FromStrRadixErr","","",16,null],[10,"from_str_radix","","Convert from a string and radix &lt;= 36.",16,{"inputs":[{"name":"str"},{"name":"u32"}],"output":{"name":"result"}}],[11,"fmt","","",1,null],[11,"fmt","","",0,null]],"paths":[[3,"ParseFloatError"],[4,"FloatErrorKind"],[8,"Zero"],[8,"One"],[8,"Signed"],[8,"Saturating"],[8,"CheckedAdd"],[8,"CheckedSub"],[8,"CheckedMul"],[8,"CheckedDiv"],[8,"Bounded"],[8,"Float"],[8,"ToPrimitive"],[8,"FromPrimitive"],[8,"NumCast"],[8,"PrimInt"],[8,"Num"]]};
initSearch(searchIndex);
