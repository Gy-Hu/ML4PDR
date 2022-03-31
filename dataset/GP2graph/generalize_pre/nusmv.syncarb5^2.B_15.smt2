; benchmark generated from python API
(set-info :status unknown)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun i4 () Bool)
(declare-fun i6 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v14 () Bool)
(declare-fun v26 () Bool)
(declare-fun v22 () Bool)
(declare-fun v30 () Bool)
(declare-fun i8 () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v28 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= v28_prime true))
(assert
 (= i10 true))
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= v20_prime false))
(assert
 (= v14_prime true))
(assert
 (= v12 false))
(assert
 (= i4 false))
(assert
 (= i6 false))
(assert
 (= v18 true))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v22_prime false))
(assert
 (= v24 false))
(assert
 (= v14 false))
(assert
 (= v26 false))
(assert
 (= v22 true))
(assert
 (= v30 true))
(assert
 (= i8 false))
(assert
 (= v24_prime false))
(assert
 (= v26_prime false))
(assert
 (= v28 true))
(assert
 (= v18_prime true))
(assert
 (= v30_prime true))
(assert
 (let (($x57 (not (and v30 (not v28)))))
 (let (($x58 (and $x57 i10)))
 (let (($x223 (and v22 v18 $x58)))
 (not $x223)))))
(check-sat)
