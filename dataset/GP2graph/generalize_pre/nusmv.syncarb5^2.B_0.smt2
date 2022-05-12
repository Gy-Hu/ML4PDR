; benchmark generated from python API
(set-info :status unknown)
(declare-fun v30_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v24 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v22 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i2_prime () Bool)
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
(assert
 (= v30_prime true))
(assert
 (= i4_prime true))
(assert
 (= v16_prime true))
(assert
 (= v20 false))
(assert
 (= i6_prime true))
(assert
 (= v12 true))
(assert
 (= v18_prime true))
(assert
 (= v18 true))
(assert
 (= v12_prime true))
(assert
 (= v16 true))
(assert
 (= v24 false))
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
 (= v14_prime true))
(assert
 (= i2_prime true))
(assert
 (= i4 true))
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
 (= i2 true))
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
 (let (($x300 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x226 (not i10_prime)))
 (let (($x190 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x354 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x339 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x216 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x184 (not (and (not (and $x216 $x339 $x354 $x190 $x300 $x226)) $x300))))
 (let (($x302 (not (and $x216 (not (and $x339 $x354 $x190 $x300))))))
 (let (($x204 (and (not (and $x216 $x339 $x354 $x190 $x300 $x226 (not i8_prime))) $x190)))
 (let (($x296 (not $x204)))
 (let (($x293 (not (and $x216 $x339 $x354 $x190 $x300 $x226 (not i8_prime) (not i6_prime)))))
 (let (($x337 (not (and $x293 $x339))))
 (let (($x259 (not i4_prime)))
 (let (($x192 (not i6_prime)))
 (let (($x172 (not i8_prime)))
 (let (($x171 (not (and (not (and $x216 $x339 $x354 $x190 $x300 $x226 $x172 $x192 $x259)) $x354))))
 (let (($x396 (and (not (and $x171 i2_prime $x337 i4_prime)) (not (and $x171 i2_prime $x296 i6_prime)) (not (and $x171 i2_prime $x184 i8_prime)) (not (and $x171 i2_prime $x302 i10_prime)) (not (and $x337 i4_prime $x296 i6_prime)) (not (and $x337 i4_prime $x184 i8_prime)) (not (and $x337 i4_prime $x302 i10_prime)) (not (and $x296 i6_prime $x184 i8_prime)) (not (and $x296 i6_prime $x302 i10_prime)) (not (and $x302 i10_prime $x184 i8_prime)))))
 (let (($x389 (not $x396)))
 (not $x389))))))))))))))))))))
(check-sat)
