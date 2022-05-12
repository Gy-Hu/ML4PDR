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
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v18 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i2 () Bool)
(declare-fun i4 () Bool)
(declare-fun i8 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun i8_prime () Bool)
(assert
 (= v30_prime true))
(assert
 (= i4_prime true))
(assert
 (= v16_prime true))
(assert
 (= v20 true))
(assert
 (= i6_prime false))
(assert
 (= v12 true))
(assert
 (= v18_prime true))
(assert
 (= v24 true))
(assert
 (= v12_prime true))
(assert
 (= v16 false))
(assert
 (= v18 true))
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
 (= v14_prime true))
(assert
 (= i2_prime true))
(assert
 (= i10 true))
(assert
 (= v22_prime false))
(assert
 (= i6 true))
(assert
 (= i2 true))
(assert
 (= i4 true))
(assert
 (= i8 true))
(assert
 (= v28_prime true))
(assert
 (= v26 false))
(assert
 (= v26_prime false))
(assert
 (= v24_prime true))
(assert
 (= v30 true))
(assert
 (= i8_prime false))
(assert
 (let (($x411 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x523 (not i10_prime)))
 (let (($x202 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x203 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x320 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x323 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x402 (not (and (not (and $x323 $x320 $x203 $x202 $x411 $x523)) $x411))))
 (let (($x449 (not (and $x323 (not (and $x320 $x203 $x202 $x411))))))
 (let (($x335 (and (not (and $x323 $x320 $x203 $x202 $x411 $x523 (not i8_prime))) $x202)))
 (let (($x220 (not $x335)))
 (let (($x328 (not (and $x323 $x320 $x203 $x202 $x411 $x523 (not i8_prime) (not i6_prime)))))
 (let (($x576 (not (and $x328 $x320))))
 (let (($x482 (not i4_prime)))
 (let (($x155 (not i6_prime)))
 (let (($x219 (not i8_prime)))
 (let (($x370 (not (and (not (and $x323 $x320 $x203 $x202 $x411 $x523 $x219 $x155 $x482)) $x203))))
 (let (($x373 (and (not (and $x370 i2_prime $x576 i4_prime)) (not (and $x370 i2_prime $x220 i6_prime)) (not (and $x370 i2_prime $x402 i8_prime)) (not (and $x370 i2_prime $x449 i10_prime)) (not (and $x576 i4_prime $x220 i6_prime)) (not (and $x576 i4_prime $x402 i8_prime)) (not (and $x576 i4_prime $x449 i10_prime)) (not (and $x220 i6_prime $x402 i8_prime)) (not (and $x220 i6_prime $x449 i10_prime)) (not (and $x449 i10_prime $x402 i8_prime)))))
 (let (($x164 (not $x373)))
 (not $x164))))))))))))))))))))
(check-sat)
