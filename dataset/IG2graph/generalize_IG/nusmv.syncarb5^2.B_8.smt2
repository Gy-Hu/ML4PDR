; benchmark generated from python API
(set-info :status unknown)
(declare-fun v12 () Bool)
(declare-fun v26 () Bool)
(declare-fun v30 () Bool)
(declare-fun v28 () Bool)
(declare-fun v24 () Bool)
(declare-fun v22 () Bool)
(declare-fun v20 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v14 () Bool)
(declare-fun i2 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12_prime () Bool)
(assert
 (= v12 true))
(assert
 (= v26 true))
(assert
 (= v30 false))
(assert
 (let (($x70 (not v30)))
 (let (($x491 (and v12 v26 $x70)))
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
 (let (($x524 (and $x141)))
 (let (($x581 (and (not (and v12_prime (not (and (not (and $x35 $x36)) i2)))) (not (and (not (and $x35 $x36)) i2 (not v12_prime))) (not (and v14_prime $x40)) (not (and v18 (not v14_prime))) (not (and v16_prime (not (and (not (and $x40 $x41)) i4)))) (not (and (not (and $x40 $x41)) i4 (not v16_prime))) (not (and v18_prime $x45)) (not (and v22 (not v18_prime))) (not (and v20_prime (not (and (not (and $x45 $x46)) i6)))) (not (and (not (and $x45 $x46)) i6 (not v20_prime))) (not (and v22_prime $x50)) (not (and v26 (not v22_prime))) (not (and v24_prime (not (and (not (and $x50 $x51)) i8)))) (not (and (not (and $x50 $x51)) i8 (not v24_prime))) (not (and v26_prime v30)) (not (and $x70 (not v26_prime))) (not (and v28_prime (not (and (not (and v30 $x55)) i10)))) (not (and (not (and v30 $x55)) i10 (not v28_prime))) (not (and v30_prime v14)) (not (and $x35 (not v30_prime))))))
 (let (($x305 (and $x524 (not $x491) $x581 (and (and (not (and $x35 $x36)) i2) $x70 (not $x35)))))
 (not (and (not $x305) (not (and $x524 $x491))))))))))))))))))))
(check-sat)
