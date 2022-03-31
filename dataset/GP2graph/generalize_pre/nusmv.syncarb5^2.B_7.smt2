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
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v18 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i2 () Bool)
(declare-fun i4 () Bool)
(declare-fun i8 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
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
 (= v12 true))
(assert
 (= v18_prime true))
(assert
 (= v24 false))
(assert
 (= v12_prime true))
(assert
 (= v16 false))
(assert
 (= v18 true))
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
 (= v14_prime true))
(assert
 (= i2_prime true))
(assert
 (= i10 true))
(assert
 (= v22_prime true))
(assert
 (= i6 true))
(assert
 (= i2 true))
(assert
 (= i4 true))
(assert
 (= i8 true))
(assert
 (= v28_prime true))
(assert
 (= v26 true))
(assert
 (= v26_prime false))
(assert
 (= v24_prime true))
(assert
 (= v30 true))
(assert
 (= i8_prime false))
(assert
 (let (($x322 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x511 (not i10_prime)))
 (let (($x567 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x224 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x574 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x391 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x354 (not (and (not (and $x391 $x574 $x224 $x567 $x322 $x511)) $x322))))
 (let (($x291 (not (and $x391 (not (and $x574 $x224 $x567 $x322))))))
 (let (($x589 (and (not (and $x391 $x574 $x224 $x567 $x322 $x511 (not i8_prime))) $x567)))
 (let (($x331 (not $x589)))
 (let (($x412 (not (and $x391 $x574 $x224 $x567 $x322 $x511 (not i8_prime) (not i6_prime)))))
 (let (($x317 (not (and $x412 $x574))))
 (let (($x231 (not i4_prime)))
 (let (($x603 (not i6_prime)))
 (let (($x178 (not i8_prime)))
 (let (($x560 (not (and (not (and $x391 $x574 $x224 $x567 $x322 $x511 $x178 $x603 $x231)) $x224))))
 (let (($x377 (and (not (and $x560 i2_prime $x317 i4_prime)) (not (and $x560 i2_prime $x331 i6_prime)) (not (and $x560 i2_prime $x354 i8_prime)) (not (and $x560 i2_prime $x291 i10_prime)) (not (and $x317 i4_prime $x331 i6_prime)) (not (and $x317 i4_prime $x354 i8_prime)) (not (and $x317 i4_prime $x291 i10_prime)) (not (and $x331 i6_prime $x354 i8_prime)) (not (and $x331 i6_prime $x291 i10_prime)) (not (and $x291 i10_prime $x354 i8_prime)))))
 (let (($x283 (not $x377)))
 (not $x283))))))))))))))))))))
(check-sat)
