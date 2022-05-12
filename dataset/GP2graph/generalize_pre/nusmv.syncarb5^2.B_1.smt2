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
(declare-fun i2 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i8 () Bool)
(declare-fun i4 () Bool)
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
 (= v20 true))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18_prime true))
(assert
 (= v24 false))
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
 (= i2 false))
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
 (= i4 true))
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
 (let (($x528 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x168 (not i10_prime)))
 (let (($x401 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x442 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x443 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x414 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x519 (not (and (not (and $x414 $x443 $x442 $x401 $x528 $x168)) $x528))))
 (let (($x306 (not (and $x414 (not (and $x443 $x442 $x401 $x528))))))
 (let (($x518 (and (not (and $x414 $x443 $x442 $x401 $x528 $x168 (not i8_prime))) $x401)))
 (let (($x481 (not $x518)))
 (let (($x454 (not (and $x414 $x443 $x442 $x401 $x528 $x168 (not i8_prime) (not i6_prime)))))
 (let (($x471 (not (and $x454 $x443))))
 (let (($x309 (not i4_prime)))
 (let (($x538 (not i6_prime)))
 (let (($x545 (not i8_prime)))
 (let (($x392 (not (and (not (and $x414 $x443 $x442 $x401 $x528 $x168 $x545 $x538 $x309)) $x442))))
 (let (($x298 (and (not (and $x392 i2_prime $x471 i4_prime)) (not (and $x392 i2_prime $x481 i6_prime)) (not (and $x392 i2_prime $x519 i8_prime)) (not (and $x392 i2_prime $x306 i10_prime)) (not (and $x471 i4_prime $x481 i6_prime)) (not (and $x471 i4_prime $x519 i8_prime)) (not (and $x471 i4_prime $x306 i10_prime)) (not (and $x481 i6_prime $x519 i8_prime)) (not (and $x481 i6_prime $x306 i10_prime)) (not (and $x306 i10_prime $x519 i8_prime)))))
 (let (($x475 (not $x298)))
 (not $x475))))))))))))))))))))
(check-sat)
