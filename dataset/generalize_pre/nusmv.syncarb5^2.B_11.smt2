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
(declare-fun v24 () Bool)
(declare-fun v16 () Bool)
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
 (= v28_prime true))
(assert
 (= v12 false))
(assert
 (= v14 true))
(assert
 (= i10_prime true))
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
 (= i4 true))
(assert
 (= i4_prime false))
(assert
 (= i2 true))
(assert
 (= v12_prime true))
(assert
 (= v26_prime false))
(assert
 (= v24 false))
(assert
 (= v16 true))
(assert
 (= i2_prime true))
(assert
 (= v26 true))
(assert
 (= v30_prime false))
(assert
 (= i8_prime false))
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
 (let (($x376 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x383 (not i10_prime)))
 (let (($x361 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x410 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x505 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x343 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x227 (not (and (not (and $x343 $x505 $x410 $x361 $x376 $x383)) $x376))))
 (let (($x168 (not (and $x343 (not (and $x505 $x410 $x361 $x376))))))
 (let (($x210 (and (not (and $x343 $x505 $x410 $x361 $x376 $x383 (not i8_prime))) $x361)))
 (let (($x482 (not $x210)))
 (let (($x574 (not (and $x343 $x505 $x410 $x361 $x376 $x383 (not i8_prime) (not i6_prime)))))
 (let (($x396 (not (and $x574 $x505))))
 (let (($x231 (not i4_prime)))
 (let (($x395 (not i6_prime)))
 (let (($x386 (not i8_prime)))
 (let (($x526 (not (and (not (and $x343 $x505 $x410 $x361 $x376 $x383 $x386 $x395 $x231)) $x410))))
 (let (($x314 (and (not (and $x526 i2_prime $x396 i4_prime)) (not (and $x526 i2_prime $x482 i6_prime)) (not (and $x526 i2_prime $x227 i8_prime)) (not (and $x526 i2_prime $x168 i10_prime)) (not (and $x396 i4_prime $x482 i6_prime)) (not (and $x396 i4_prime $x227 i8_prime)) (not (and $x396 i4_prime $x168 i10_prime)) (not (and $x482 i6_prime $x227 i8_prime)) (not (and $x482 i6_prime $x168 i10_prime)) (not (and $x168 i10_prime $x227 i8_prime)))))
 (not (not $x314))))))))))))))))))))
(check-sat)
