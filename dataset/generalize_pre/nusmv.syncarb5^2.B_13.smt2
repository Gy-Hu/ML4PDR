; benchmark generated from python API
(set-info :status unknown)
(declare-fun v28_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v14 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i6 () Bool)
(declare-fun i10 () Bool)
(declare-fun i4 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(assert
 (= v28_prime true))
(assert
 (= v12 false))
(assert
 (= v14 true))
(assert
 (= i10_prime true))
(assert
 (= v20_prime true))
(assert
 (= v20 true))
(assert
 (= v22_prime false))
(assert
 (= v24_prime true))
(assert
 (= v16_prime true))
(assert
 (= i6_prime false))
(assert
 (= v30 true))
(assert
 (= i6 true))
(assert
 (= i10 true))
(assert
 (= i4 true))
(assert
 (= i4_prime false))
(assert
 (= i2 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime false))
(assert
 (= v24 true))
(assert
 (= v16 true))
(assert
 (= i2_prime true))
(assert
 (= v26 false))
(assert
 (= v30_prime false))
(assert
 (= i8_prime false))
(assert
 (= v18_prime false))
(assert
 (= v18 true))
(assert
 (= v14_prime true))
(assert
 (= i8 true))
(assert
 (= v22 false))
(assert
 (= v28 true))
(assert
 (let (($x456 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x251 (not i10_prime)))
 (let (($x441 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x437 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x503 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x389 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x279 (not (and (not (and $x389 $x503 $x437 $x441 $x456 $x251)) $x456))))
 (let (($x429 (not (and $x389 (not (and $x503 $x437 $x441 $x456))))))
 (let (($x326 (and (not (and $x389 $x503 $x437 $x441 $x456 $x251 (not i8_prime))) $x441)))
 (let (($x192 (not $x326)))
 (let (($x463 (not (and $x389 $x503 $x437 $x441 $x456 $x251 (not i8_prime) (not i6_prime)))))
 (let (($x306 (not (and $x463 $x503))))
 (let (($x179 (not i4_prime)))
 (let (($x500 (not i6_prime)))
 (let (($x538 (not i8_prime)))
 (let (($x333 (not (and (not (and $x389 $x503 $x437 $x441 $x456 $x251 $x538 $x500 $x179)) $x437))))
 (let (($x452 (and (not (and $x333 i2_prime $x306 i4_prime)) (not (and $x333 i2_prime $x192 i6_prime)) (not (and $x333 i2_prime $x279 i8_prime)) (not (and $x333 i2_prime $x429 i10_prime)) (not (and $x306 i4_prime $x192 i6_prime)) (not (and $x306 i4_prime $x279 i8_prime)) (not (and $x306 i4_prime $x429 i10_prime)) (not (and $x192 i6_prime $x279 i8_prime)) (not (and $x192 i6_prime $x429 i10_prime)) (not (and $x429 i10_prime $x279 i8_prime)))))
 (not (not $x452))))))))))))))))))))
(check-sat)
