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
 (= v20_prime false))
(assert
 (= v20 true))
(assert
 (= v22_prime false))
(assert
 (= v24_prime false))
(assert
 (= v16_prime true))
(assert
 (= i6_prime false))
(assert
 (= v30 true))
(assert
 (= i6 false))
(assert
 (= i10 true))
(assert
 (= i4 true))
(assert
 (= i4_prime true))
(assert
 (= i2 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime false))
(assert
 (= v24 false))
(assert
 (= v16 true))
(assert
 (= i2_prime false))
(assert
 (= v26 false))
(assert
 (= v30_prime false))
(assert
 (= i8_prime false))
(assert
 (= v18_prime true))
(assert
 (= v18 false))
(assert
 (= v14_prime false))
(assert
 (= i8 false))
(assert
 (= v22 true))
(assert
 (= v28 true))
(assert
 (let (($x477 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x179 (not i10_prime)))
 (let (($x365 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x308 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x295 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x176 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x328 (not (and (not (and $x176 $x295 $x308 $x365 $x477 $x179)) $x477))))
 (let (($x361 (not (and $x176 (not (and $x295 $x308 $x365 $x477))))))
 (let (($x449 (and (not (and $x176 $x295 $x308 $x365 $x477 $x179 (not i8_prime))) $x365)))
 (let (($x327 (not $x449)))
 (let (($x433 (not (and $x176 $x295 $x308 $x365 $x477 $x179 (not i8_prime) (not i6_prime)))))
 (let (($x373 (not (and $x433 $x295))))
 (let (($x231 (not i4_prime)))
 (let (($x333 (not i6_prime)))
 (let (($x228 (not i8_prime)))
 (let (($x187 (not (and (not (and $x176 $x295 $x308 $x365 $x477 $x179 $x228 $x333 $x231)) $x308))))
 (let (($x498 (and (not (and $x187 i2_prime $x373 i4_prime)) (not (and $x187 i2_prime $x327 i6_prime)) (not (and $x187 i2_prime $x328 i8_prime)) (not (and $x187 i2_prime $x361 i10_prime)) (not (and $x373 i4_prime $x327 i6_prime)) (not (and $x373 i4_prime $x328 i8_prime)) (not (and $x373 i4_prime $x361 i10_prime)) (not (and $x327 i6_prime $x328 i8_prime)) (not (and $x327 i6_prime $x361 i10_prime)) (not (and $x361 i10_prime $x328 i8_prime)))))
 (not (not $x498))))))))))))))))))))
(check-sat)
