; benchmark generated from python API
(set-info :status unknown)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun i10 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v24 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v14 () Bool)
(declare-fun v26 () Bool)
(declare-fun v22 () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= v20_prime false))
(assert
 (= v14_prime false))
(assert
 (= v12 true))
(assert
 (= i10 false))
(assert
 (= v22_prime true))
(assert
 (= v18 false))
(assert
 (= v16 false))
(assert
 (= v24 false))
(assert
 (= v18_prime false))
(assert
 (= i8 false))
(assert
 (= v14 false))
(assert
 (= v26 true))
(assert
 (= v22 false))
(assert
 (= v24_prime false))
(assert
 (= v30 false))
(assert
 (= v28_prime false))
(assert
 (= v12_prime true))
(assert
 (= v26_prime true))
(assert
 (= i2 true))
(assert
 (= v30_prime true))
(assert
 (let (($x70 (not v30)))
 (let (($x38 (not (and (not v14) (not v12)))))
 (let (($x39 (and $x38 i2)))
 (let (($x186 (and $x39 $x70 v26)))
 (not $x186))))))
(check-sat)
