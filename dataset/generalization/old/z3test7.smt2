; benchmark generated from python API
(set-info :status unknown)
; (set-option :produce-unsat-cores true)
(declare-fun i8_prime () Bool)
(declare-fun p0 () Bool)
(declare-fun v12 () Bool)
(declare-fun p1 () Bool)
(declare-fun v14 () Bool)
(declare-fun p2 () Bool)
(declare-fun v20 () Bool)
(declare-fun p3 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun p4 () Bool)
(declare-fun i6 () Bool)
(declare-fun p5 () Bool)
(declare-fun v30 () Bool)
(declare-fun p6 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun p7 () Bool)
(declare-fun i10 () Bool)
(declare-fun p8 () Bool)
(declare-fun i4 () Bool)
(declare-fun p9 () Bool)
(declare-fun i2 () Bool)
(declare-fun p10 () Bool)
(declare-fun v16 () Bool)
(declare-fun p11 () Bool)
(declare-fun v24 () Bool)
(declare-fun p12 () Bool)
(declare-fun v26 () Bool)
(declare-fun p13 () Bool)
(declare-fun v18 () Bool)
(declare-fun p14 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun p15 () Bool)
(declare-fun i8 () Bool)
(declare-fun p16 () Bool)
(declare-fun v22 () Bool)
(declare-fun p17 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun p18 () Bool)
(declare-fun v28 () Bool)
(declare-fun p19 () Bool)
(assert
 (let (($x383 (= i8_prime true)))
 (=> p0 $x383)))
(assert
 (let (($x384 (= v12 true)))
 (=> p1 $x384)))
(assert
 (let (($x385 (= v14 true)))
 (=> p2 $x385)))
(assert
 (let (($x386 (= v20 false)))
 (=> p3 $x386)))
(assert
 (let (($x387 (= i2_prime true)))
 (=> p4 $x387)))
(assert
 (let (($x378 (= i6 true)))
 (=> p5 $x378)))
(assert
 (let (($x379 (= v30 false)))
 (=> p6 $x379)))
(assert
 (let (($x380 (= i4_prime true)))
 (=> p7 $x380)))
(assert
 (let (($x381 (= i10 true)))
 (=> p8 $x381)))
(assert
 (let (($x382 (= i4 true)))
 (=> p9 $x382)))
(assert
 (let (($x373 (= i2 true)))
 (=> p10 $x373)))
(assert
 (let (($x374 (= v16 false)))
 (=> p11 $x374)))
(assert
 (let (($x375 (= v24 false)))
 (=> p12 $x375)))
(assert
 (let (($x376 (= v26 true)))
 (=> p13 $x376)))
(assert
 (let (($x377 (= v18 true)))
 (=> p14 $x377)))
(assert
 (let (($x368 (= i10_prime true)))
 (=> p15 $x368)))
(assert
 (let (($x369 (= i8 true)))
 (=> p16 $x369)))
(assert
 (let (($x370 (= v22 true)))
 (=> p17 $x370)))
(assert
 (let (($x371 (= i6_prime true)))
 (=> p18 $x371)))
(assert
 (let (($x372 (= v28 false)))
 (=> p19 $x372)))
(assert
 (let (($x350 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x222 (not i10_prime)))
 (let (($x355 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x363 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x367 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x354 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x325 (not (and (not (and $x354 $x367 $x363 $x355 $x350 $x222)) $x350))))
 (let (($x348 (not (and $x354 (not (and $x367 $x363 $x355 $x350))))))
 (let (($x360 (not (and (not (and $x354 $x367 $x363 $x355 $x350 $x222 (not i8_prime))) $x355))))
 (let (($x339 (and (not (and $x354 $x367 $x363 $x355 $x350 $x222 (not i8_prime) (not i6_prime))) $x367)))
 (let (($x323 (not $x339)))
 (let (($x227 (not i4_prime)))
 (let (($x174 (not i6_prime)))
 (let (($x186 (not i8_prime)))
 (let (($x342 (not (and (not (and $x354 $x367 $x363 $x355 $x350 $x222 $x186 $x174 $x227)) $x363))))
 (let (($x155 (and (not (and $x342 i2_prime $x323 i4_prime)) (not (and $x342 i2_prime $x360 i6_prime)) (not (and $x342 i2_prime $x325 i8_prime)) (not (and $x342 i2_prime $x348 i10_prime)) (not (and $x323 i4_prime $x360 i6_prime)) (not (and $x323 i4_prime $x325 i8_prime)) (not (and $x323 i4_prime $x348 i10_prime)) (not (and $x360 i6_prime $x325 i8_prime)) (not (and $x360 i6_prime $x348 i10_prime)) (not (and $x348 i10_prime $x325 i8_prime)))))
 (not (not $x155)))))))))))))))))))
(check-sat)
;(check-sat-assuming (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19))
;(get-unsat-core)
