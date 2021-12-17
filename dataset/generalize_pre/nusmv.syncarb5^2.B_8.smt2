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
(declare-fun i2 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
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
 (= v16_prime true))
(assert
 (= i6_prime true))
(assert
 (= v30 true))
(assert
 (= i6 true))
(assert
 (= i10 true))
(assert
 (= i2 true))
(assert
 (= i4_prime false))
(assert
 (= i4 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime false))
(assert
 (= v24 true))
(assert
 (= v16 false))
(assert
 (= i2_prime true))
(assert
 (= v26 true))
(assert
 (= i8_prime false))
(assert
 (= v30_prime true))
(assert
 (= v18_prime false))
(assert
 (= v18 true))
(assert
 (= v14_prime true))
(assert
 (= i8 true))
(assert
 (= v22 false))
(assert
 (= v28 true))
(assert
 (let (($x378 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x228 (not i10_prime)))
 (let (($x254 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x300 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x313 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x386 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x334 (not (and (not (and $x386 $x313 $x300 $x254 $x378 $x228)) $x378))))
 (let (($x282 (not (and $x386 (not (and $x313 $x300 $x254 $x378))))))
 (let (($x156 (and (not (and $x386 $x313 $x300 $x254 $x378 $x228 (not i8_prime))) $x254)))
 (let (($x172 (not $x156)))
 (let (($x482 (not (and $x386 $x313 $x300 $x254 $x378 $x228 (not i8_prime) (not i6_prime)))))
 (let (($x349 (not (and $x482 $x313))))
 (let (($x170 (not i4_prime)))
 (let (($x361 (not i6_prime)))
 (let (($x340 (not i8_prime)))
 (let (($x241 (not (and (not (and $x386 $x313 $x300 $x254 $x378 $x228 $x340 $x361 $x170)) $x300))))
 (let (($x454 (and (not (and $x241 i2_prime $x349 i4_prime)) (not (and $x241 i2_prime $x172 i6_prime)) (not (and $x241 i2_prime $x334 i8_prime)) (not (and $x241 i2_prime $x282 i10_prime)) (not (and $x349 i4_prime $x172 i6_prime)) (not (and $x349 i4_prime $x334 i8_prime)) (not (and $x349 i4_prime $x282 i10_prime)) (not (and $x172 i6_prime $x334 i8_prime)) (not (and $x172 i6_prime $x282 i10_prime)) (not (and $x282 i10_prime $x334 i8_prime)))))
 (not (not $x454))))))))))))))))))))
(check-sat)
