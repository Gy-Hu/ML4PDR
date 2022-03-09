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
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 true))
(assert
 (= v18_prime true))
(assert
 (= v24 true))
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
 (let (($x491 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x204 (not i10_prime)))
 (let (($x529 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x162 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x465 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x210 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x376 (not (and (not (and $x210 $x465 $x162 $x529 $x491 $x204)) $x491))))
 (let (($x339 (not (and $x210 (not (and $x465 $x162 $x529 $x491))))))
 (let (($x566 (and (not (and $x210 $x465 $x162 $x529 $x491 $x204 (not i8_prime))) $x529)))
 (let (($x267 (not $x566)))
 (let (($x587 (not (and $x210 $x465 $x162 $x529 $x491 $x204 (not i8_prime) (not i6_prime)))))
 (let (($x360 (not (and $x587 $x465))))
 (let (($x610 (not i4_prime)))
 (let (($x523 (not i6_prime)))
 (let (($x466 (not i8_prime)))
 (let (($x611 (not (and (not (and $x210 $x465 $x162 $x529 $x491 $x204 $x466 $x523 $x610)) $x162))))
 (let (($x255 (and (not (and $x611 i2_prime $x360 i4_prime)) (not (and $x611 i2_prime $x267 i6_prime)) (not (and $x611 i2_prime $x376 i8_prime)) (not (and $x611 i2_prime $x339 i10_prime)) (not (and $x360 i4_prime $x267 i6_prime)) (not (and $x360 i4_prime $x376 i8_prime)) (not (and $x360 i4_prime $x339 i10_prime)) (not (and $x267 i6_prime $x376 i8_prime)) (not (and $x267 i6_prime $x339 i10_prime)) (not (and $x339 i10_prime $x376 i8_prime)))))
 (let (($x603 (not $x255)))
 (not $x603))))))))))))))))))))
(check-sat)
