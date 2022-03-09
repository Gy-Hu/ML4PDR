; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18 () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i6 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i8 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime true))
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
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v18_prime true))
(assert
 (= v14 false))
(assert
 (= v28 true))
(assert
 (= v22 true))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i10 false))
(assert
 (= i6 true))
(assert
 (= v22_prime true))
(assert
 (= v28_prime false))
(assert
 (= i2 false))
(assert
 (= i8 true))
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
 (let (($x449 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x580 (not i10_prime)))
 (let (($x231 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x366 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x492 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x434 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x173 (not (and (not (and $x434 $x492 $x366 $x231 $x449 $x580)) $x449))))
 (let (($x455 (not (and $x434 (not (and $x492 $x366 $x231 $x449))))))
 (let (($x361 (and (not (and $x434 $x492 $x366 $x231 $x449 $x580 (not i8_prime))) $x231)))
 (let (($x330 (not $x361)))
 (let (($x416 (not (and $x434 $x492 $x366 $x231 $x449 $x580 (not i8_prime) (not i6_prime)))))
 (let (($x245 (not (and $x416 $x492))))
 (let (($x383 (not i4_prime)))
 (let (($x590 (not i6_prime)))
 (let (($x599 (not i8_prime)))
 (let (($x296 (not (and (not (and $x434 $x492 $x366 $x231 $x449 $x580 $x599 $x590 $x383)) $x366))))
 (let (($x378 (and (not (and $x296 i2_prime $x245 i4_prime)) (not (and $x296 i2_prime $x330 i6_prime)) (not (and $x296 i2_prime $x173 i8_prime)) (not (and $x296 i2_prime $x455 i10_prime)) (not (and $x245 i4_prime $x330 i6_prime)) (not (and $x245 i4_prime $x173 i8_prime)) (not (and $x245 i4_prime $x455 i10_prime)) (not (and $x330 i6_prime $x173 i8_prime)) (not (and $x330 i6_prime $x455 i10_prime)) (not (and $x455 i10_prime $x173 i8_prime)))))
 (let (($x476 (not $x378)))
 (not $x476))))))))))))))))))))
(check-sat)
