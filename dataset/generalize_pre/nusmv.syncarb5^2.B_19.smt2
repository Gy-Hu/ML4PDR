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
(declare-fun v16_prime () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i6 () Bool)
(declare-fun i10 () Bool)
(declare-fun i4 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v24 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(assert
 (= v28_prime false))
(assert
 (= v12 false))
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
 (= v16_prime false))
(assert
 (= i6_prime true))
(assert
 (= v30 false))
(assert
 (= i6 true))
(assert
 (= i10 false))
(assert
 (= i4 false))
(assert
 (= i4_prime false))
(assert
 (= i2 false))
(assert
 (= v12_prime false))
(assert
 (= v26_prime true))
(assert
 (= v16 false))
(assert
 (= v24 true))
(assert
 (= i2_prime false))
(assert
 (= v26 true))
(assert
 (= v30_prime true))
(assert
 (= i8_prime true))
(assert
 (= v18_prime false))
(assert
 (= v18 false))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 false))
(assert
 (= v28 false))
(assert
 (let (($x200 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x230 (not i10_prime)))
 (let (($x499 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x497 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x462 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x251 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x323 (not (and (not (and $x251 $x462 $x497 $x499 $x200 $x230)) $x200))))
 (let (($x359 (not (and $x251 (not (and $x462 $x497 $x499 $x200))))))
 (let (($x328 (and (not (and $x251 $x462 $x497 $x499 $x200 $x230 (not i8_prime))) $x499)))
 (let (($x295 (not $x328)))
 (let (($x303 (not (and $x251 $x462 $x497 $x499 $x200 $x230 (not i8_prime) (not i6_prime)))))
 (let (($x436 (not (and $x303 $x462))))
 (let (($x170 (not i4_prime)))
 (let (($x476 (not i6_prime)))
 (let (($x404 (not i8_prime)))
 (let (($x405 (not (and (not (and $x251 $x462 $x497 $x499 $x200 $x230 $x404 $x476 $x170)) $x497))))
 (let (($x277 (and (not (and $x405 i2_prime $x436 i4_prime)) (not (and $x405 i2_prime $x295 i6_prime)) (not (and $x405 i2_prime $x323 i8_prime)) (not (and $x405 i2_prime $x359 i10_prime)) (not (and $x436 i4_prime $x295 i6_prime)) (not (and $x436 i4_prime $x323 i8_prime)) (not (and $x436 i4_prime $x359 i10_prime)) (not (and $x295 i6_prime $x323 i8_prime)) (not (and $x295 i6_prime $x359 i10_prime)) (not (and $x359 i10_prime $x323 i8_prime)))))
 (not (not $x277))))))))))))))))))))
(check-sat)
