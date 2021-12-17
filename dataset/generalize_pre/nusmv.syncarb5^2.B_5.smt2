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
(declare-fun v26 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun i2_prime () Bool)
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
 (= v20 false))
(assert
 (= v22_prime true))
(assert
 (= v24_prime true))
(assert
 (= v16_prime true))
(assert
 (= i6_prime true))
(assert
 (= v30 false))
(assert
 (= i6 true))
(assert
 (= i10 false))
(assert
 (= i2 false))
(assert
 (= i4_prime true))
(assert
 (= i4 true))
(assert
 (= v12_prime false))
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
 (= v18_prime true))
(assert
 (= v18 false))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 true))
(assert
 (= v28 true))
(assert
 (let (($x198 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x369 (not i10_prime)))
 (let (($x201 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x221 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x186 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x370 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x219 (not (and (not (and $x370 $x186 $x221 $x201 $x198 $x369)) $x198))))
 (let (($x237 (not (and $x370 (not (and $x186 $x221 $x201 $x198))))))
 (let (($x426 (and (not (and $x370 $x186 $x221 $x201 $x198 $x369 (not i8_prime))) $x201)))
 (let (($x333 (not $x426)))
 (let (($x325 (not (and $x370 $x186 $x221 $x201 $x198 $x369 (not i8_prime) (not i6_prime)))))
 (let (($x178 (not (and $x325 $x186))))
 (let (($x323 (not i4_prime)))
 (let (($x336 (not i6_prime)))
 (let (($x173 (not i8_prime)))
 (let (($x164 (not (and (not (and $x370 $x186 $x221 $x201 $x198 $x369 $x173 $x336 $x323)) $x221))))
 (let (($x355 (and (not (and $x164 i2_prime $x178 i4_prime)) (not (and $x164 i2_prime $x333 i6_prime)) (not (and $x164 i2_prime $x219 i8_prime)) (not (and $x164 i2_prime $x237 i10_prime)) (not (and $x178 i4_prime $x333 i6_prime)) (not (and $x178 i4_prime $x219 i8_prime)) (not (and $x178 i4_prime $x237 i10_prime)) (not (and $x333 i6_prime $x219 i8_prime)) (not (and $x333 i6_prime $x237 i10_prime)) (not (and $x237 i10_prime $x219 i8_prime)))))
 (not (not $x355))))))))))))))))))))
(check-sat)
