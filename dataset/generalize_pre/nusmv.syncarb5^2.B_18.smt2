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
(declare-fun v16 () Bool)
(declare-fun v24 () Bool)
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
 (= v12 true))
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
 (= v30 false))
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
 (= v26_prime true))
(assert
 (= v16 true))
(assert
 (= v24 true))
(assert
 (= i2_prime false))
(assert
 (= v26 false))
(assert
 (= v30_prime false))
(assert
 (= i8_prime true))
(assert
 (= v18_prime false))
(assert
 (= v18 false))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 false))
(assert
 (= v28 false))
(assert
 (let (($x448 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x154 (not i10_prime)))
 (let (($x506 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x176 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x337 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x293 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x445 (not (and (not (and $x293 $x337 $x176 $x506 $x448 $x154)) $x448))))
 (let (($x558 (not (and $x293 (not (and $x337 $x176 $x506 $x448))))))
 (let (($x297 (and (not (and $x293 $x337 $x176 $x506 $x448 $x154 (not i8_prime))) $x506)))
 (let (($x509 (not $x297)))
 (let (($x325 (not (and $x293 $x337 $x176 $x506 $x448 $x154 (not i8_prime) (not i6_prime)))))
 (let (($x575 (not (and $x325 $x337))))
 (let (($x453 (not i4_prime)))
 (let (($x311 (not i6_prime)))
 (let (($x365 (not i8_prime)))
 (let (($x463 (not (and (not (and $x293 $x337 $x176 $x506 $x448 $x154 $x365 $x311 $x453)) $x176))))
 (let (($x438 (and (not (and $x463 i2_prime $x575 i4_prime)) (not (and $x463 i2_prime $x509 i6_prime)) (not (and $x463 i2_prime $x445 i8_prime)) (not (and $x463 i2_prime $x558 i10_prime)) (not (and $x575 i4_prime $x509 i6_prime)) (not (and $x575 i4_prime $x445 i8_prime)) (not (and $x575 i4_prime $x558 i10_prime)) (not (and $x509 i6_prime $x445 i8_prime)) (not (and $x509 i6_prime $x558 i10_prime)) (not (and $x558 i10_prime $x445 i8_prime)))))
 (not (not $x438))))))))))))))))))))
(check-sat)
