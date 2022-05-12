; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
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
(declare-fun v30_prime () Bool)
(assert
 (= i8_prime true))
(assert
 (= i4_prime true))
(assert
 (= v16_prime true))
(assert
 (= v20 false))
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
 (= v18 false))
(assert
 (= v16 true))
(assert
 (= v14 false))
(assert
 (= v28 false))
(assert
 (= v22 true))
(assert
 (= i10_prime true))
(assert
 (= v20_prime true))
(assert
 (= v14_prime false))
(assert
 (= i2_prime false))
(assert
 (= i4 true))
(assert
 (= v22_prime false))
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
 (= v26 false))
(assert
 (= v26_prime true))
(assert
 (= v24_prime true))
(assert
 (= v30 false))
(assert
 (= v30_prime true))
(assert
 (let (($x486 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x703 (not i10_prime)))
 (let (($x415 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x522 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x686 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x286 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x419 (not (and (not (and $x286 $x686 $x522 $x415 $x486 $x703)) $x486))))
 (let (($x647 (not (and $x286 (not (and $x686 $x522 $x415 $x486))))))
 (let (($x405 (and (not (and $x286 $x686 $x522 $x415 $x486 $x703 (not i8_prime))) $x415)))
 (let (($x173 (not $x405)))
 (let (($x472 (not (and $x286 $x686 $x522 $x415 $x486 $x703 (not i8_prime) (not i6_prime)))))
 (let (($x189 (not (and $x472 $x686))))
 (let (($x624 (not i4_prime)))
 (let (($x471 (not i6_prime)))
 (let (($x373 (not i8_prime)))
 (let (($x220 (not (and (not (and $x286 $x686 $x522 $x415 $x486 $x703 $x373 $x471 $x624)) $x522))))
 (let (($x337 (and (not (and $x220 i2_prime $x189 i4_prime)) (not (and $x220 i2_prime $x173 i6_prime)) (not (and $x220 i2_prime $x419 i8_prime)) (not (and $x220 i2_prime $x647 i10_prime)) (not (and $x189 i4_prime $x173 i6_prime)) (not (and $x189 i4_prime $x419 i8_prime)) (not (and $x189 i4_prime $x647 i10_prime)) (not (and $x173 i6_prime $x419 i8_prime)) (not (and $x173 i6_prime $x647 i10_prime)) (not (and $x647 i10_prime $x419 i8_prime)))))
 (let (($x342 (not $x337)))
 (not $x342))))))))))))))))))))
(check-sat)
