; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
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
(declare-fun i4 () Bool)
(declare-fun i6 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i8 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun i6_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= i8_prime true))
(assert
 (= v16_prime false))
(assert
 (= v20 false))
(assert
 (= v12 false))
(assert
 (= v18 false))
(assert
 (= v24 true))
(assert
 (= v12_prime false))
(assert
 (= v16 false))
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
 (= v20_prime false))
(assert
 (= v14_prime false))
(assert
 (= i2 false))
(assert
 (= i4 false))
(assert
 (= i6 false))
(assert
 (= v28_prime true))
(assert
 (= i10 true))
(assert
 (= i8 true))
(assert
 (= v22_prime true))
(assert
 (= v26 true))
(assert
 (= v26_prime true))
(assert
 (= v30 false))
(assert
 (= v24_prime true))
(assert
 (= v30_prime false))
(assert
 (let (($x585 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x303 (not i10_prime)))
 (let (($x444 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x413 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x325 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x344 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x178 (not (and (not (and $x344 $x325 $x413 $x444 $x585 $x303)) $x585))))
 (let (($x569 (not (and $x344 (not (and $x325 $x413 $x444 $x585))))))
 (let (($x210 (and (not (and $x344 $x325 $x413 $x444 $x585 $x303 (not i8_prime))) $x444)))
 (let (($x470 (not $x210)))
 (let (($x211 (not (and $x344 $x325 $x413 $x444 $x585 $x303 (not i8_prime) (not i6_prime)))))
 (let (($x452 (not (and $x211 $x325))))
 (let (($x164 (not i4_prime)))
 (let (($x437 (not i6_prime)))
 (let (($x231 (not i8_prime)))
 (let (($x223 (not (and (not (and $x344 $x325 $x413 $x444 $x585 $x303 $x231 $x437 $x164)) $x413))))
 (let (($x313 (and (not (and $x223 i2_prime $x452 i4_prime)) (not (and $x223 i2_prime $x470 i6_prime)) (not (and $x223 i2_prime $x178 i8_prime)) (not (and $x223 i2_prime $x569 i10_prime)) (not (and $x452 i4_prime $x470 i6_prime)) (not (and $x452 i4_prime $x178 i8_prime)) (not (and $x452 i4_prime $x569 i10_prime)) (not (and $x470 i6_prime $x178 i8_prime)) (not (and $x470 i6_prime $x569 i10_prime)) (not (and $x569 i10_prime $x178 i8_prime)))))
 (let (($x316 (not $x313)))
 (not $x316))))))))))))))))))))
(check-sat)
