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
(declare-fun i2 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i4 () Bool)
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
(declare-fun i2_prime () Bool)
(assert
 (= v28_prime true))
(assert
 (= v12 true))
(assert
 (= v14 false))
(assert
 (= i10_prime false))
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
 (= i2 true))
(assert
 (= i4_prime true))
(assert
 (= i4 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime true))
(assert
 (= v24 true))
(assert
 (= v16 true))
(assert
 (= v26 false))
(assert
 (= i8_prime true))
(assert
 (= v30_prime true))
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
 (let (($x227 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x191 (not i10_prime)))
 (let (($x354 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x172 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x177 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x316 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x311 (not (and (not (and $x316 $x177 $x172 $x354 $x227 $x191)) $x227))))
 (let (($x318 (not (and $x316 (not (and $x177 $x172 $x354 $x227))))))
 (let (($x277 (and (not (and $x316 $x177 $x172 $x354 $x227 $x191 (not i8_prime))) $x354)))
 (let (($x418 (not $x277)))
 (let (($x378 (not (and $x316 $x177 $x172 $x354 $x227 $x191 (not i8_prime) (not i6_prime)))))
 (let (($x165 (not (and $x378 $x177))))
 (let (($x200 (not i4_prime)))
 (let (($x241 (not i6_prime)))
 (let (($x283 (not i8_prime)))
 (let (($x439 (not (and (not (and $x316 $x177 $x172 $x354 $x227 $x191 $x283 $x241 $x200)) $x172))))
 (let (($x422 (and (not (and $x439 i2_prime $x165 i4_prime)) (not (and $x439 i2_prime $x418 i6_prime)) (not (and $x439 i2_prime $x311 i8_prime)) (not (and $x439 i2_prime $x318 i10_prime)) (not (and $x165 i4_prime $x418 i6_prime)) (not (and $x165 i4_prime $x311 i8_prime)) (not (and $x165 i4_prime $x318 i10_prime)) (not (and $x418 i6_prime $x311 i8_prime)) (not (and $x418 i6_prime $x318 i10_prime)) (not (and $x318 i10_prime $x311 i8_prime)))))
 (not (not $x422))))))))))))))))))))
(check-sat)
