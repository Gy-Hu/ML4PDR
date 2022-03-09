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
(declare-fun v12_prime () Bool)
(declare-fun v18 () Bool)
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
 (= v12_prime false))
(assert
 (= v18 true))
(assert
 (= v14 false))
(assert
 (= v28 false))
(assert
 (= v22 false))
(assert
 (= i10_prime false))
(assert
 (= v20_prime true))
(assert
 (= v14_prime true))
(assert
 (= i10 false))
(assert
 (= v22_prime true))
(assert
 (= i6 true))
(assert
 (= i4 false))
(assert
 (= v28_prime false))
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
 (let (($x326 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x197 (not i10_prime)))
 (let (($x155 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x488 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x530 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x188 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x332 (not (and (not (and $x188 $x530 $x488 $x155 $x326 $x197)) $x326))))
 (let (($x552 (not (and $x188 (not (and $x530 $x488 $x155 $x326))))))
 (let (($x577 (and (not (and $x188 $x530 $x488 $x155 $x326 $x197 (not i8_prime))) $x155)))
 (let (($x450 (not $x577)))
 (let (($x485 (not (and $x188 $x530 $x488 $x155 $x326 $x197 (not i8_prime) (not i6_prime)))))
 (let (($x447 (not (and $x485 $x530))))
 (let (($x158 (not i4_prime)))
 (let (($x405 (not i6_prime)))
 (let (($x466 (not i8_prime)))
 (let (($x199 (not (and (not (and $x188 $x530 $x488 $x155 $x326 $x197 $x466 $x405 $x158)) $x488))))
 (let (($x520 (and (not (and $x199 i2_prime $x447 i4_prime)) (not (and $x199 i2_prime $x450 i6_prime)) (not (and $x199 i2_prime $x332 i8_prime)) (not (and $x199 i2_prime $x552 i10_prime)) (not (and $x447 i4_prime $x450 i6_prime)) (not (and $x447 i4_prime $x332 i8_prime)) (not (and $x447 i4_prime $x552 i10_prime)) (not (and $x450 i6_prime $x332 i8_prime)) (not (and $x450 i6_prime $x552 i10_prime)) (not (and $x552 i10_prime $x332 i8_prime)))))
 (let (($x562 (not $x520)))
 (not $x562))))))))))))))))))))
(check-sat)
