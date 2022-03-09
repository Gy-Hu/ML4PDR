; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v18 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i4 () Bool)
(declare-fun i6 () Bool)
(declare-fun i8 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime true))
(assert
 (= i4_prime true))
(assert
 (= v16_prime true))
(assert
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18_prime true))
(assert
 (= v24 true))
(assert
 (= v18 false))
(assert
 (= v12_prime false))
(assert
 (= v16 true))
(assert
 (= v14 false))
(assert
 (= v28 false))
(assert
 (= v22 true))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i10 true))
(assert
 (= v22_prime true))
(assert
 (= i2 false))
(assert
 (= i4 true))
(assert
 (= i6 true))
(assert
 (= i8 true))
(assert
 (= v28_prime true))
(assert
 (= v26 true))
(assert
 (= v26_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= i8_prime true))
(assert
 (let (($x291 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x576 (not i10_prime)))
 (let (($x179 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x312 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x574 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x551 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x298 (not (and (not (and $x551 $x574 $x312 $x179 $x291 $x576)) $x291))))
 (let (($x446 (not (and $x551 (not (and $x574 $x312 $x179 $x291))))))
 (let (($x393 (and (not (and $x551 $x574 $x312 $x179 $x291 $x576 (not i8_prime))) $x179)))
 (let (($x305 (not $x393)))
 (let (($x406 (not (and $x551 $x574 $x312 $x179 $x291 $x576 (not i8_prime) (not i6_prime)))))
 (let (($x480 (not (and $x406 $x574))))
 (let (($x490 (not i4_prime)))
 (let (($x483 (not i6_prime)))
 (let (($x458 (not i8_prime)))
 (let (($x323 (not (and (not (and $x551 $x574 $x312 $x179 $x291 $x576 $x458 $x483 $x490)) $x312))))
 (let (($x308 (and (not (and $x323 i2_prime $x480 i4_prime)) (not (and $x323 i2_prime $x305 i6_prime)) (not (and $x323 i2_prime $x298 i8_prime)) (not (and $x323 i2_prime $x446 i10_prime)) (not (and $x480 i4_prime $x305 i6_prime)) (not (and $x480 i4_prime $x298 i8_prime)) (not (and $x480 i4_prime $x446 i10_prime)) (not (and $x305 i6_prime $x298 i8_prime)) (not (and $x305 i6_prime $x446 i10_prime)) (not (and $x446 i10_prime $x298 i8_prime)))))
 (let (($x494 (not $x308)))
 (not $x494))))))))))))))))))))
(check-sat)
