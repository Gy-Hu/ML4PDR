; benchmark generated from python API
(set-info :status unknown)
(declare-fun v28 () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v14 () Bool)
(declare-fun v26 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v30 () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v22 () Bool)
(declare-fun i4 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(assert
 (= v28 true))
(assert
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v28_prime true))
(assert
 (= v12 false))
(assert
 (= v14 false))
(assert
 (= v26 true))
(assert
 (= v20_prime true))
(assert
 (= v18 true))
(assert
 (= v18_prime false))
(assert
 (= v20 true))
(assert
 (= v30_prime true))
(assert
 (= i6 true))
(assert
 (= v30 true))
(assert
 (= v24_prime false))
(assert
 (= v14_prime true))
(assert
 (= i8 false))
(assert
 (= v16_prime true))
(assert
 (= v22_prime true))
(assert
 (= i10 true))
(assert
 (= v22 false))
(assert
 (= i4 true))
(assert
 (= v12_prime false))
(assert
 (= v26_prime false))
(assert
 (let (($x58 (and (not (and v30 (not v28))) i10)))
 (let (($x44 (and (not (and (not v18) (not v16))) i4)))
 (not (and v18 $x44 v26 $x58)))))
(check-sat)
