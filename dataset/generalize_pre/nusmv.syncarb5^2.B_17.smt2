; benchmark generated from python API
(set-info :status unknown)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v14 () Bool)
(declare-fun v26 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i10 () Bool)
(declare-fun v22 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v28 () Bool)
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
 (= v26 false))
(assert
 (= v20_prime false))
(assert
 (= v18 false))
(assert
 (= v30_prime true))
(assert
 (= v20 false))
(assert
 (= v18_prime true))
(assert
 (= i6 false))
(assert
 (= v22_prime false))
(assert
 (= v16_prime false))
(assert
 (= v14_prime false))
(assert
 (= v24_prime false))
(assert
 (= v30 false))
(assert
 (= i10 true))
(assert
 (= v22 true))
(assert
 (= v12_prime false))
(assert
 (= v26_prime true))
(assert
 (not (and (and (not (and v30 (not v28))) i10) (not v30) v22)))
(check-sat)
