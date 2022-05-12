; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8 () Bool)
(declare-fun i4 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun i2 () Bool)
(declare-fun i10 () Bool)
(declare-fun v18 () Bool)
(declare-fun v24 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun v30 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= i8 true))
(assert
 (= i4 true))
(assert
 (= v16_prime true))
(assert
 (= v20 false))
(assert
 (= v20_prime false))
(assert
 (= v14_prime false))
(assert
 (= v12 false))
(assert
 (= i2 false))
(assert
 (= i10 true))
(assert
 (= v18 false))
(assert
 (= v24 false))
(assert
 (= v12_prime false))
(assert
 (= v16 true))
(assert
 (= v18_prime false))
(assert
 (= v26 true))
(assert
 (= v28 false))
(assert
 (= v22 false))
(assert
 (= v30 true))
(assert
 (= v26_prime false))
(assert
 (= v28_prime false))
(assert
 (= v14 true))
(assert
 (= v24_prime true))
(assert
 (= v22_prime true))
(assert
 (= v30_prime false))
(assert
 (let (($x43 (not (and (not v18) (not v16)))))
 (let (($x44 (and $x43 i4)))
 (let (($x53 (not (and (not v26) (not v24)))))
 (let (($x54 (and $x53 i8)))
 (let (($x422 (and $x54 $x44 v26 (not (not v14)))))
 (not $x422)))))))
(check-sat)
