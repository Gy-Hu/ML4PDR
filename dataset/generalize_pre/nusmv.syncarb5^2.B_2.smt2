; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
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
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime false))
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18 false))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v24 true))
(assert
 (= v18_prime false))
(assert
 (= v14 true))
(assert
 (= v28 false))
(assert
 (= v22 false))
(assert
 (= i10_prime true))
(assert
 (= v20_prime false))
(assert
 (= v14_prime false))
(assert
 (= i4 false))
(assert
 (= v22_prime true))
(assert
 (= i6 false))
(assert
 (= v28_prime true))
(assert
 (= i10 true))
(assert
 (= i8 true))
(assert
 (= i2 false))
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
 (let (($x235 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x479 (not i10_prime)))
 (let (($x221 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x354 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x337 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x574 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x368 (not (and (not (and $x574 $x337 $x354 $x221 $x235 $x479)) $x235))))
 (let (($x378 (not (and $x574 (not (and $x337 $x354 $x221 $x235))))))
 (let (($x189 (and (not (and $x574 $x337 $x354 $x221 $x235 $x479 (not i8_prime))) $x221)))
 (let (($x502 (not $x189)))
 (let (($x581 (not (and $x574 $x337 $x354 $x221 $x235 $x479 (not i8_prime) (not i6_prime)))))
 (let (($x205 (not (and $x581 $x337))))
 (let (($x441 (not i4_prime)))
 (let (($x460 (not i6_prime)))
 (let (($x226 (not i8_prime)))
 (let (($x179 (not (and (not (and $x574 $x337 $x354 $x221 $x235 $x479 $x226 $x460 $x441)) $x354))))
 (let (($x259 (and (not (and $x179 i2_prime $x205 i4_prime)) (not (and $x179 i2_prime $x502 i6_prime)) (not (and $x179 i2_prime $x368 i8_prime)) (not (and $x179 i2_prime $x378 i10_prime)) (not (and $x205 i4_prime $x502 i6_prime)) (not (and $x205 i4_prime $x368 i8_prime)) (not (and $x205 i4_prime $x378 i10_prime)) (not (and $x502 i6_prime $x368 i8_prime)) (not (and $x502 i6_prime $x378 i10_prime)) (not (and $x378 i10_prime $x368 i8_prime)))))
 (let (($x397 (not $x259)))
 (not $x397))))))))))))))))))))
(check-sat)
