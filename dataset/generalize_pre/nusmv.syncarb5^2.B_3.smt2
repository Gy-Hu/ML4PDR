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
(declare-fun v30 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i2 () Bool)
(declare-fun i4 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v26 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun i4_prime () Bool)
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
 (= v22_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= v16_prime true))
(assert
 (= i6 true))
(assert
 (= i6_prime true))
(assert
 (= i10 true))
(assert
 (= i2 true))
(assert
 (= i4 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime true))
(assert
 (= v24 false))
(assert
 (= v16 true))
(assert
 (= v26 true))
(assert
 (= i8_prime true))
(assert
 (= v30_prime true))
(assert
 (= v18 false))
(assert
 (= v18_prime false))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 false))
(assert
 (= v28 true))
(assert
 (let (($x228 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x191 (not i10_prime)))
 (let (($x447 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x373 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x375 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x336 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x178 (not (and (not (and $x336 $x375 $x373 $x447 $x228 $x191)) $x228))))
 (let (($x316 (not (and $x336 (not (and $x375 $x373 $x447 $x228))))))
 (let (($x172 (and (not (and $x336 $x375 $x373 $x447 $x228 $x191 (not i8_prime))) $x447)))
 (let (($x339 (not $x172)))
 (let (($x214 (not (and $x336 $x375 $x373 $x447 $x228 $x191 (not i8_prime) (not i6_prime)))))
 (let (($x354 (not (and $x214 $x375))))
 (let (($x215 (not i4_prime)))
 (let (($x365 (not i6_prime)))
 (let (($x171 (not i8_prime)))
 (let (($x154 (not (and (not (and $x336 $x375 $x373 $x447 $x228 $x191 $x171 $x365 $x215)) $x373))))
 (let (($x328 (and (not (and $x154 i2_prime $x354 i4_prime)) (not (and $x154 i2_prime $x339 i6_prime)) (not (and $x154 i2_prime $x178 i8_prime)) (not (and $x154 i2_prime $x316 i10_prime)) (not (and $x354 i4_prime $x339 i6_prime)) (not (and $x354 i4_prime $x178 i8_prime)) (not (and $x354 i4_prime $x316 i10_prime)) (not (and $x339 i6_prime $x178 i8_prime)) (not (and $x339 i6_prime $x316 i10_prime)) (not (and $x316 i10_prime $x178 i8_prime)))))
 (not (not $x328))))))))))))))))))))
(check-sat)
