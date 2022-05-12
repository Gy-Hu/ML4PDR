; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i8 () Bool)
(declare-fun i2 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= i8_prime true))
(assert
 (= i4_prime false))
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18_prime false))
(assert
 (= v24 true))
(assert
 (= v12_prime false))
(assert
 (= v18 false))
(assert
 (= v16 true))
(assert
 (= v14 true))
(assert
 (= v28 true))
(assert
 (= v22 false))
(assert
 (= i10_prime true))
(assert
 (= v20_prime false))
(assert
 (= v14_prime false))
(assert
 (= i2_prime true))
(assert
 (= i4 false))
(assert
 (= v22_prime false))
(assert
 (= i6 true))
(assert
 (= v28_prime true))
(assert
 (= i10 true))
(assert
 (= i8 true))
(assert
 (= i2 false))
(assert
 (= v26 false))
(assert
 (= v26_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= v30_prime false))
(assert
 (let (($x225 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x443 (not i10_prime)))
 (let (($x199 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x558 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x178 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x286 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x500 (not (and (not (and $x286 $x178 $x558 $x199 $x225 $x443)) $x225))))
 (let (($x509 (not (and $x286 (not (and $x178 $x558 $x199 $x225))))))
 (let (($x694 (and (not (and $x286 $x178 $x558 $x199 $x225 $x443 (not i8_prime))) $x199)))
 (let (($x465 (not $x694)))
 (let (($x464 (not (and $x286 $x178 $x558 $x199 $x225 $x443 (not i8_prime) (not i6_prime)))))
 (let (($x160 (not (and $x464 $x178))))
 (let (($x364 (not i4_prime)))
 (let (($x505 (not i6_prime)))
 (let (($x461 (not i8_prime)))
 (let (($x354 (not (and (not (and $x286 $x178 $x558 $x199 $x225 $x443 $x461 $x505 $x364)) $x558))))
 (let (($x280 (and (not (and $x354 i2_prime $x160 i4_prime)) (not (and $x354 i2_prime $x465 i6_prime)) (not (and $x354 i2_prime $x500 i8_prime)) (not (and $x354 i2_prime $x509 i10_prime)) (not (and $x160 i4_prime $x465 i6_prime)) (not (and $x160 i4_prime $x500 i8_prime)) (not (and $x160 i4_prime $x509 i10_prime)) (not (and $x465 i6_prime $x500 i8_prime)) (not (and $x465 i6_prime $x509 i10_prime)) (not (and $x509 i10_prime $x500 i8_prime)))))
 (let (($x326 (not $x280)))
 (not $x326))))))))))))))))))))
(check-sat)
