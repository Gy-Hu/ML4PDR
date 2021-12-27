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
(declare-fun v26 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun i2_prime () Bool)
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
 (= v24 false))
(assert
 (= v16 true))
(assert
 (= v26 true))
(assert
 (= v30_prime false))
(assert
 (= i8_prime true))
(assert
 (= v18_prime true))
(assert
 (= v18 false))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 true))
(assert
 (= v28 false))
(assert
 (let (($x275 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x222 (not i10_prime)))
 (let (($x390 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x354 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x439 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x316 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x462 (not (and (not (and $x316 $x439 $x354 $x390 $x275 $x222)) $x275))))
 (let (($x394 (not (and $x316 (not (and $x439 $x354 $x390 $x275))))))
 (let (($x178 (and (not (and $x316 $x439 $x354 $x390 $x275 $x222 (not i8_prime))) $x390)))
 (let (($x153 (not $x178)))
 (let (($x426 (not (and $x316 $x439 $x354 $x390 $x275 $x222 (not i8_prime) (not i6_prime)))))
 (let (($x418 (not (and $x426 $x439))))
 (let (($x184 (not i4_prime)))
 (let (($x382 (not i6_prime)))
 (let (($x379 (not i8_prime)))
 (let (($x359 (not (and (not (and $x316 $x439 $x354 $x390 $x275 $x222 $x379 $x382 $x184)) $x354))))
 (let (($x454 (and (not (and $x359 i2_prime $x418 i4_prime)) (not (and $x359 i2_prime $x153 i6_prime)) (not (and $x359 i2_prime $x462 i8_prime)) (not (and $x359 i2_prime $x394 i10_prime)) (not (and $x418 i4_prime $x153 i6_prime)) (not (and $x418 i4_prime $x462 i8_prime)) (not (and $x418 i4_prime $x394 i10_prime)) (not (and $x153 i6_prime $x462 i8_prime)) (not (and $x153 i6_prime $x394 i10_prime)) (not (and $x394 i10_prime $x462 i8_prime)))))
 (not (not $x454))))))))))))))))))))
(check-sat)
