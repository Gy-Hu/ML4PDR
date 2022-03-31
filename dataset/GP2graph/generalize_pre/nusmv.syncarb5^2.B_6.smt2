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
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v18_prime true))
(assert
 (= v14 false))
(assert
 (= v28 true))
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
 (let (($x448 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x535 (not i10_prime)))
 (let (($x406 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x420 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x499 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x310 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x364 (not (and (not (and $x310 $x499 $x420 $x406 $x448 $x535)) $x448))))
 (let (($x472 (not (and $x310 (not (and $x499 $x420 $x406 $x448))))))
 (let (($x451 (and (not (and $x310 $x499 $x420 $x406 $x448 $x535 (not i8_prime))) $x406)))
 (let (($x204 (not $x451)))
 (let (($x449 (not (and $x310 $x499 $x420 $x406 $x448 $x535 (not i8_prime) (not i6_prime)))))
 (let (($x526 (not (and $x449 $x499))))
 (let (($x205 (not i4_prime)))
 (let (($x510 (not i6_prime)))
 (let (($x525 (not i8_prime)))
 (let (($x424 (not (and (not (and $x310 $x499 $x420 $x406 $x448 $x535 $x525 $x510 $x205)) $x420))))
 (let (($x431 (and (not (and $x424 i2_prime $x526 i4_prime)) (not (and $x424 i2_prime $x204 i6_prime)) (not (and $x424 i2_prime $x364 i8_prime)) (not (and $x424 i2_prime $x472 i10_prime)) (not (and $x526 i4_prime $x204 i6_prime)) (not (and $x526 i4_prime $x364 i8_prime)) (not (and $x526 i4_prime $x472 i10_prime)) (not (and $x204 i6_prime $x364 i8_prime)) (not (and $x204 i6_prime $x472 i10_prime)) (not (and $x472 i10_prime $x364 i8_prime)))))
 (let (($x557 (not $x431)))
 (not $x557))))))))))))))))))))
(check-sat)
