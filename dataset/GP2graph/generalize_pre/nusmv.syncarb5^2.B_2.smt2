; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
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
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime false))
(assert
 (= v16_prime false))
(assert
 (= v20 true))
(assert
 (= i6_prime true))
(assert
 (= v12 true))
(assert
 (= v18 false))
(assert
 (= v16 false))
(assert
 (= v12_prime false))
(assert
 (= v24 false))
(assert
 (= v18_prime false))
(assert
 (= v14 true))
(assert
 (= v28 false))
(assert
 (= v22 false))
(assert
 (= i10_prime true))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i4 false))
(assert
 (= v22_prime true))
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
 (let (($x171 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x553 (not i10_prime)))
 (let (($x512 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x396 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x155 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x297 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x446 (not (and (not (and $x297 $x155 $x396 $x512 $x171 $x553)) $x171))))
 (let (($x182 (not (and $x297 (not (and $x155 $x396 $x512 $x171))))))
 (let (($x479 (and (not (and $x297 $x155 $x396 $x512 $x171 $x553 (not i8_prime))) $x512)))
 (let (($x509 (not $x479)))
 (let (($x510 (not (and $x297 $x155 $x396 $x512 $x171 $x553 (not i8_prime) (not i6_prime)))))
 (let (($x308 (not (and $x510 $x155))))
 (let (($x428 (not i4_prime)))
 (let (($x461 (not i6_prime)))
 (let (($x403 (not i8_prime)))
 (let (($x581 (not (and (not (and $x297 $x155 $x396 $x512 $x171 $x553 $x403 $x461 $x428)) $x396))))
 (let (($x473 (and (not (and $x581 i2_prime $x308 i4_prime)) (not (and $x581 i2_prime $x509 i6_prime)) (not (and $x581 i2_prime $x446 i8_prime)) (not (and $x581 i2_prime $x182 i10_prime)) (not (and $x308 i4_prime $x509 i6_prime)) (not (and $x308 i4_prime $x446 i8_prime)) (not (and $x308 i4_prime $x182 i10_prime)) (not (and $x509 i6_prime $x446 i8_prime)) (not (and $x509 i6_prime $x182 i10_prime)) (not (and $x182 i10_prime $x446 i8_prime)))))
 (let (($x153 (not $x473)))
 (not $x153))))))))))))))))))))
(check-sat)
