; benchmark generated from python API
(set-info :status unknown)
(set-logic ALL)
; (set-option :dump-unsat-cores-full true)
(set-option :produce-unsat-cores true)
(declare-fun i8_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v14 () Bool)
(declare-fun v20 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v30 () Bool)
(declare-fun i4_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun i4 () Bool)
(declare-fun i2 () Bool)
(declare-fun v16 () Bool)
(declare-fun v24 () Bool)
(declare-fun v26 () Bool)
(declare-fun v18 () Bool)
(declare-fun i10_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v28 () Bool)
(assert
 (= i8_prime true))
(assert
 (= v12 true))
(assert
 (= v14 true))
(assert
 (= v20 false))
(assert
 (= i2_prime true))
(assert
 (= i6 true))
(assert
 (= v30 false))
(assert
 (= i4_prime true))
(assert
 (= i10 true))
(assert
 (= i4 true))
(assert
 (= i2 true))
(assert
 (= v16 false))
(assert
 (= v24 false))
(assert
 (= v26 true))
(assert
 (= v18 true))
(assert
 (= i10_prime true))
(assert
 (= i8 true))
(assert
 (= v22 true))
(assert
 (= i6_prime true))
(assert
 (= v28 false))
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
(get-unsat-core)
