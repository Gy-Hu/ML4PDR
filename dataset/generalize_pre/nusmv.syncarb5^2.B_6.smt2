; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18 () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i6 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i8 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= i8_prime true))
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18 false))
(assert
 (= v24 true))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v18_prime true))
(assert
 (= v14 false))
(assert
 (= v28 false))
(assert
 (= v22 true))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i10 false))
(assert
 (= i6 true))
(assert
 (= v22_prime true))
(assert
 (= v28_prime false))
(assert
 (= i2 false))
(assert
 (= i8 true))
(assert
 (= v26 true))
(assert
 (= v26_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= v30_prime true))
(assert
 (let (($x390 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x212 (not i10_prime)))
 (let (($x444 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x453 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x434 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x520 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x487 (not (and (not (and $x520 $x434 $x453 $x444 $x390 $x212)) $x390))))
 (let (($x457 (not (and $x520 (not (and $x434 $x453 $x444 $x390))))))
 (let (($x302 (and (not (and $x520 $x434 $x453 $x444 $x390 $x212 (not i8_prime))) $x444)))
 (let (($x294 (not $x302)))
 (let (($x328 (not (and $x520 $x434 $x453 $x444 $x390 $x212 (not i8_prime) (not i6_prime)))))
 (let (($x321 (not (and $x328 $x434))))
 (let (($x451 (not i4_prime)))
 (let (($x498 (not i6_prime)))
 (let (($x353 (not i8_prime)))
 (let (($x450 (not (and (not (and $x520 $x434 $x453 $x444 $x390 $x212 $x353 $x498 $x451)) $x453))))
 (let (($x327 (and (not (and $x450 i2_prime $x321 i4_prime)) (not (and $x450 i2_prime $x294 i6_prime)) (not (and $x450 i2_prime $x487 i8_prime)) (not (and $x450 i2_prime $x457 i10_prime)) (not (and $x321 i4_prime $x294 i6_prime)) (not (and $x321 i4_prime $x487 i8_prime)) (not (and $x321 i4_prime $x457 i10_prime)) (not (and $x294 i6_prime $x487 i8_prime)) (not (and $x294 i6_prime $x457 i10_prime)) (not (and $x457 i10_prime $x487 i8_prime)))))
 (let (($x560 (not $x327)))
 (not $x560))))))))))))))))))))
(check-sat)
