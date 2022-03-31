; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i8 () Bool)
(declare-fun i2 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= i8_prime false))
(assert
 (= i4_prime false))
(assert
 (= v16_prime true))
(assert
 (= v20 true))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18_prime false))
(assert
 (= v24 true))
(assert
 (= v12_prime true))
(assert
 (= v18 false))
(assert
 (= v16 true))
(assert
 (= v14 true))
(assert
 (= v28 true))
(assert
 (= v22 false))
(assert
 (= i10_prime true))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i2_prime false))
(assert
 (= i4 true))
(assert
 (= v22_prime true))
(assert
 (= i6 true))
(assert
 (= v28_prime true))
(assert
 (= i10 true))
(assert
 (= i8 true))
(assert
 (= i2 true))
(assert
 (= v26 true))
(assert
 (= v26_prime false))
(assert
 (= v24_prime true))
(assert
 (= v30 true))
(assert
 (= v30_prime false))
(assert
 (let (($x549 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x338 (not i10_prime)))
 (let (($x210 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x234 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x157 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x414 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x619 (not (and (not (and $x414 $x157 $x234 $x210 $x549 $x338)) $x549))))
 (let (($x581 (not (and $x414 (not (and $x157 $x234 $x210 $x549))))))
 (let (($x384 (and (not (and $x414 $x157 $x234 $x210 $x549 $x338 (not i8_prime))) $x210)))
 (let (($x393 (not $x384)))
 (let (($x597 (not (and $x414 $x157 $x234 $x210 $x549 $x338 (not i8_prime) (not i6_prime)))))
 (let (($x500 (not (and $x597 $x157))))
 (let (($x482 (not i4_prime)))
 (let (($x365 (not i6_prime)))
 (let (($x472 (not i8_prime)))
 (let (($x623 (not (and (not (and $x414 $x157 $x234 $x210 $x549 $x338 $x472 $x365 $x482)) $x234))))
 (let (($x160 (and (not (and $x623 i2_prime $x500 i4_prime)) (not (and $x623 i2_prime $x393 i6_prime)) (not (and $x623 i2_prime $x619 i8_prime)) (not (and $x623 i2_prime $x581 i10_prime)) (not (and $x500 i4_prime $x393 i6_prime)) (not (and $x500 i4_prime $x619 i8_prime)) (not (and $x500 i4_prime $x581 i10_prime)) (not (and $x393 i6_prime $x619 i8_prime)) (not (and $x393 i6_prime $x581 i10_prime)) (not (and $x581 i10_prime $x619 i8_prime)))))
 (let (($x182 (not $x160)))
 (not $x182))))))))))))))))))))
(check-sat)
