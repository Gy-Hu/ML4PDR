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
(declare-fun v30 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i2 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
(declare-fun v26 () Bool)
(declare-fun i8_prime () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun i4 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i2_prime () Bool)
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
 (= v20 false))
(assert
 (= v22_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= v16_prime false))
(assert
 (= i6 true))
(assert
 (= i6_prime true))
(assert
 (= i10 true))
(assert
 (= i2 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime true))
(assert
 (= v24 false))
(assert
 (= v16 false))
(assert
 (= v26 true))
(assert
 (= i8_prime true))
(assert
 (= v30_prime true))
(assert
 (= v18 false))
(assert
 (= v18_prime true))
(assert
 (= v14_prime false))
(assert
 (= i8 true))
(assert
 (= v22 true))
(assert
 (= v28 true))
(assert
 (let (($x172 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x208 (not i10_prime)))
 (let (($x156 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x299 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x309 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x275 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x215 (not (and (not (and $x275 $x309 $x299 $x156 $x172 $x208)) $x172))))
 (let (($x387 (not (and $x275 (not (and $x309 $x299 $x156 $x172))))))
 (let (($x379 (and (not (and $x275 $x309 $x299 $x156 $x172 $x208 (not i8_prime))) $x156)))
 (let (($x328 (not $x379)))
 (let (($x180 (not (and $x275 $x309 $x299 $x156 $x172 $x208 (not i8_prime) (not i6_prime)))))
 (let (($x203 (not (and $x180 $x309))))
 (let (($x287 (not i4_prime)))
 (let (($x153 (not i6_prime)))
 (let (($x349 (not i8_prime)))
 (let (($x191 (not (and (not (and $x275 $x309 $x299 $x156 $x172 $x208 $x349 $x153 $x287)) $x299))))
 (let (($x339 (and (not (and $x191 i2_prime $x203 i4_prime)) (not (and $x191 i2_prime $x328 i6_prime)) (not (and $x191 i2_prime $x215 i8_prime)) (not (and $x191 i2_prime $x387 i10_prime)) (not (and $x203 i4_prime $x328 i6_prime)) (not (and $x203 i4_prime $x215 i8_prime)) (not (and $x203 i4_prime $x387 i10_prime)) (not (and $x328 i6_prime $x215 i8_prime)) (not (and $x328 i6_prime $x387 i10_prime)) (not (and $x387 i10_prime $x215 i8_prime)))))
 (not (not $x339))))))))))))))))))))
(check-sat)
