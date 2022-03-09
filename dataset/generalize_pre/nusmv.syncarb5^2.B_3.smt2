; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18 () Bool)
(declare-fun v24 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i10 () Bool)
(declare-fun i4 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i8 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime true))
(assert
 (= v16_prime false))
(assert
 (= v20 true))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18 false))
(assert
 (= v24 false))
(assert
 (= v12_prime false))
(assert
 (= v16 false))
(assert
 (= v18_prime false))
(assert
 (= v14 false))
(assert
 (= v28 true))
(assert
 (= v22 false))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i2 false))
(assert
 (= i10 false))
(assert
 (= i4 false))
(assert
 (= v28_prime false))
(assert
 (= i6 true))
(assert
 (= i8 true))
(assert
 (= v22_prime true))
(assert
 (= v26 true))
(assert
 (= v26_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= i8_prime true))
(assert
 (let (($x569 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x463 (not i10_prime)))
 (let (($x390 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x485 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x434 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x324 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x153 (not (and (not (and $x324 $x434 $x485 $x390 $x569 $x463)) $x569))))
 (let (($x328 (not (and $x324 (not (and $x434 $x485 $x390 $x569))))))
 (let (($x453 (and (not (and $x324 $x434 $x485 $x390 $x569 $x463 (not i8_prime))) $x390)))
 (let (($x176 (not $x453)))
 (let (($x186 (not (and $x324 $x434 $x485 $x390 $x569 $x463 (not i8_prime) (not i6_prime)))))
 (let (($x447 (not (and $x186 $x434))))
 (let (($x439 (not i4_prime)))
 (let (($x375 (not i6_prime)))
 (let (($x290 (not i8_prime)))
 (let (($x162 (not (and (not (and $x324 $x434 $x485 $x390 $x569 $x463 $x290 $x375 $x439)) $x485))))
 (let (($x536 (and (not (and $x162 i2_prime $x447 i4_prime)) (not (and $x162 i2_prime $x176 i6_prime)) (not (and $x162 i2_prime $x153 i8_prime)) (not (and $x162 i2_prime $x328 i10_prime)) (not (and $x447 i4_prime $x176 i6_prime)) (not (and $x447 i4_prime $x153 i8_prime)) (not (and $x447 i4_prime $x328 i10_prime)) (not (and $x176 i6_prime $x153 i8_prime)) (not (and $x176 i6_prime $x328 i10_prime)) (not (and $x328 i10_prime $x153 i8_prime)))))
 (let (($x199 (not $x536)))
 (not $x199))))))))))))))))))))
(check-sat)
