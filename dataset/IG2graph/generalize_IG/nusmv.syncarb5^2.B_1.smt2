; benchmark generated from python API
(set-info :status unknown)
(declare-fun v20 () Bool)
(declare-fun v16 () Bool)
(declare-fun v22 () Bool)
(declare-fun v26 () Bool)
(declare-fun v30 () Bool)
(declare-fun v28 () Bool)
(declare-fun v24 () Bool)
(declare-fun v18 () Bool)
(declare-fun v14 () Bool)
(declare-fun v12 () Bool)
(declare-fun i4 () Bool)
(declare-fun i6 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12_prime () Bool)
(declare-fun i2 () Bool)
(assert
 (= v20 true))
(assert
 (= v16 true))
(assert
 (= v22 true))
(assert
 (= v26 true))
(assert
 (let (($x468 (and v20 v16 v22 v26)))
 (let (($x70 (not v30)))
 (let (($x55 (not v28)))
 (let (($x50 (not v26)))
 (let (($x51 (not v24)))
 (let (($x45 (not v22)))
 (let (($x46 (not v20)))
 (let (($x40 (not v18)))
 (let (($x41 (not v16)))
 (let (($x35 (not v14)))
 (let (($x36 (not v12)))
 (let (($x141 (and $x36 $x35 $x41 $x40 $x46 $x45 $x51 $x50 $x55 $x70)))
 (let (($x231 (and $x141)))
 (let (($x43 (not (and $x40 $x41))))
 (let (($x44 (and $x43 i4)))
 (let (($x48 (not (and $x45 $x46))))
 (let (($x49 (and $x48 i6)))
 (let (($x401 (and (not (and v12_prime (not (and (not (and $x35 $x36)) i2)))) (not (and (not (and $x35 $x36)) i2 (not v12_prime))) (not (and v14_prime $x40)) (not (and v18 (not v14_prime))) (not (and v16_prime (not $x44))) (not (and $x43 i4 (not v16_prime))) (not (and v18_prime $x45)) (not (and v22 (not v18_prime))) (not (and v20_prime (not $x49))) (not (and $x48 i6 (not v20_prime))) (not (and v22_prime $x50)) (not (and v26 (not v22_prime))) (not (and v24_prime (not (and (not (and $x50 $x51)) i8)))) (not (and (not (and $x50 $x51)) i8 (not v24_prime))) (not (and v26_prime v30)) (not (and $x70 (not v26_prime))) (not (and v28_prime (not (and (not (and v30 $x55)) i10)))) (not (and (not (and v30 $x55)) i10 (not v28_prime))) (not (and v30_prime v14)) (not (and $x35 (not v30_prime))))))
 (let (($x581 (and (not (and $x231 (not $x468) $x401 (and $x49 $x44 v26 $x70))) (not (and $x231 $x468)))))
 (not $x581)))))))))))))))))))))
(check-sat)
