; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v18 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i4 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i2_prime () Bool)
(assert
 (= v30_prime true))
(assert
 (= i4_prime false))
(assert
 (= v16_prime false))
(assert
 (= v20 true))
(assert
 (= i6_prime true))
(assert
 (= v12 false))
(assert
 (= v18_prime false))
(assert
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v18 false))
(assert
 (= v12_prime false))
(assert
 (= v14 false))
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
 (= i10 true))
(assert
 (= v22_prime true))
(assert
 (= i6 true))
(assert
 (= i4 false))
(assert
 (= v28_prime true))
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
 (= i8_prime true))
(assert
 (let (($x493 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x166 (not i10_prime)))
 (let (($x489 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x370 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x218 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x459 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x161 (not (and (not (and $x459 $x218 $x370 $x489 $x493 $x166)) $x493))))
 (let (($x517 (not (and $x459 (not (and $x218 $x370 $x489 $x493))))))
 (let (($x284 (and (not (and $x459 $x218 $x370 $x489 $x493 $x166 (not i8_prime))) $x489)))
 (let (($x186 (not $x284)))
 (let (($x486 (not (and $x459 $x218 $x370 $x489 $x493 $x166 (not i8_prime) (not i6_prime)))))
 (let (($x461 (not (and $x486 $x218))))
 (let (($x358 (not i4_prime)))
 (let (($x335 (not i6_prime)))
 (let (($x386 (not i8_prime)))
 (let (($x371 (not (and (not (and $x459 $x218 $x370 $x489 $x493 $x166 $x386 $x335 $x358)) $x370))))
 (let (($x177 (and (not (and $x371 i2_prime $x461 i4_prime)) (not (and $x371 i2_prime $x186 i6_prime)) (not (and $x371 i2_prime $x161 i8_prime)) (not (and $x371 i2_prime $x517 i10_prime)) (not (and $x461 i4_prime $x186 i6_prime)) (not (and $x461 i4_prime $x161 i8_prime)) (not (and $x461 i4_prime $x517 i10_prime)) (not (and $x186 i6_prime $x161 i8_prime)) (not (and $x186 i6_prime $x517 i10_prime)) (not (and $x517 i10_prime $x161 i8_prime)))))
 (let (($x468 (not $x177)))
 (not $x468))))))))))))))))))))
(check-sat)
