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
 (= v20 false))
(assert
 (= v22_prime true))
(assert
 (= v24_prime true))
(assert
 (= v16_prime true))
(assert
 (= i6_prime true))
(assert
 (= v30 false))
(assert
 (= i6 true))
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
 (= v26_prime true))
(assert
 (= v16 false))
(assert
 (= v24 false))
(assert
 (= i2_prime true))
(assert
 (= v26 true))
(assert
 (= v30_prime false))
(assert
 (= i8_prime true))
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
 (= v28 false))
(assert
 (let (($x333 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x222 (not i10_prime)))
 (let (($x361 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x324 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x331 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x347 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x351 (not (and (not (and $x347 $x331 $x324 $x361 $x333 $x222)) $x333))))
 (let (($x345 (not (and $x347 (not (and $x331 $x324 $x361 $x333))))))
 (let (($x352 (and (not (and $x347 $x331 $x324 $x361 $x333 $x222 (not i8_prime))) $x361)))
 (let (($x336 (not $x352)))
 (let (($x349 (not (and $x347 $x331 $x324 $x361 $x333 $x222 (not i8_prime) (not i6_prime)))))
 (let (($x354 (not (and $x349 $x331))))
 (let (($x227 (not i4_prime)))
 (let (($x174 (not i6_prime)))
 (let (($x186 (not i8_prime)))
 (let (($x211 (not (and (not (and $x347 $x331 $x324 $x361 $x333 $x222 $x186 $x174 $x227)) $x324))))
 (let (($x206 (and (not (and $x211 i2_prime $x354 i4_prime)) (not (and $x211 i2_prime $x336 i6_prime)) (not (and $x211 i2_prime $x351 i8_prime)) (not (and $x211 i2_prime $x345 i10_prime)) (not (and $x354 i4_prime $x336 i6_prime)) (not (and $x354 i4_prime $x351 i8_prime)) (not (and $x354 i4_prime $x345 i10_prime)) (not (and $x336 i6_prime $x351 i8_prime)) (not (and $x336 i6_prime $x345 i10_prime)) (not (and $x345 i10_prime $x351 i8_prime)))))
 (not (not $x206))))))))))))))))))))
(check-sat)
