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
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v26 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun i2 () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v28_prime true))
(assert
 (= v12 false))
(assert
 (= v14 false))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v20 true))
(assert
 (= v22_prime true))
(assert
 (= v24_prime true))
(assert
 (= v16_prime true))
(assert
 (= i6_prime true))
(assert
 (= v30 true))
(assert
 (= i6 true))
(assert
 (= i10 true))
(assert
 (= i4 true))
(assert
 (= i4_prime true))
(assert
 (= v12_prime false))
(assert
 (= v26_prime false))
(assert
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v26 true))
(assert
 (= i8_prime false))
(assert
 (= v30_prime true))
(assert
 (= v18_prime true))
(assert
 (= v18 true))
(assert
 (= v14_prime true))
(assert
 (= i8 true))
(assert
 (= v22 true))
(assert
 (= v28 true))
(assert
 (let (($x283 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x444 (not i10_prime)))
 (let (($x276 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x538 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x496 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x333 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x445 (not (and (not (and $x333 $x496 $x538 $x276 $x283 $x444)) $x283))))
 (let (($x530 (not (and $x333 (not (and $x496 $x538 $x276 $x283))))))
 (let (($x546 (and (not (and $x333 $x496 $x538 $x276 $x283 $x444 (not i8_prime))) $x276)))
 (let (($x219 (not $x546)))
 (let (($x382 (not (and $x333 $x496 $x538 $x276 $x283 $x444 (not i8_prime) (not i6_prime)))))
 (let (($x196 (not (and $x382 $x496))))
 (let (($x452 (not i4_prime)))
 (let (($x334 (not i6_prime)))
 (let (($x181 (not i8_prime)))
 (let (($x179 (not (and (not (and $x333 $x496 $x538 $x276 $x283 $x444 $x181 $x334 $x452)) $x538))))
 (let (($x227 (and (not (and $x179 i2_prime $x196 i4_prime)) (not (and $x179 i2_prime $x219 i6_prime)) (not (and $x179 i2_prime $x445 i8_prime)) (not (and $x179 i2_prime $x530 i10_prime)) (not (and $x196 i4_prime $x219 i6_prime)) (not (and $x196 i4_prime $x445 i8_prime)) (not (and $x196 i4_prime $x530 i10_prime)) (not (and $x219 i6_prime $x445 i8_prime)) (not (and $x219 i6_prime $x530 i10_prime)) (not (and $x530 i10_prime $x445 i8_prime)))))
 (not (not $x227))))))))))))))))))))
(check-sat)
