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
 (= v30_prime false))
(assert
 (= i4_prime false))
(assert
 (= v16_prime true))
(assert
 (= v20 true))
(assert
 (= i6_prime false))
(assert
 (= v12 true))
(assert
 (= v18_prime false))
(assert
 (= v18 false))
(assert
 (= v12_prime true))
(assert
 (= v16 true))
(assert
 (= v24 true))
(assert
 (= v14 true))
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
 (= i8_prime true))
(assert
 (let (($x286 (not (and (not v30) (and (not (and (not v26) (not v24))) i8)))))
 (let (($x649 (not i10_prime)))
 (let (($x155 (not (and v26 (and (not (and (not v22) (not v20))) i6)))))
 (let (($x209 (not (and v18 (and (not (and (not v14) (not v12))) i2)))))
 (let (($x166 (not (and v22 (and (not (and (not v18) (not v16))) i4)))))
 (let (($x331 (not (and (not (not v14)) (and (not (and v30 (not v28))) i10)))))
 (let (($x714 (not (and (not (and $x331 $x166 $x209 $x155 $x286 $x649)) $x286))))
 (let (($x186 (not (and $x331 (not (and $x166 $x209 $x155 $x286))))))
 (let (($x701 (and (not (and $x331 $x166 $x209 $x155 $x286 $x649 (not i8_prime))) $x155)))
 (let (($x297 (not $x701)))
 (let (($x505 (not (and $x331 $x166 $x209 $x155 $x286 $x649 (not i8_prime) (not i6_prime)))))
 (let (($x618 (not (and $x505 $x166))))
 (let (($x583 (not i4_prime)))
 (let (($x690 (not i6_prime)))
 (let (($x637 (not i8_prime)))
 (let (($x517 (not (and (not (and $x331 $x166 $x209 $x155 $x286 $x649 $x637 $x690 $x583)) $x209))))
 (let (($x557 (and (not (and $x517 i2_prime $x618 i4_prime)) (not (and $x517 i2_prime $x297 i6_prime)) (not (and $x517 i2_prime $x714 i8_prime)) (not (and $x517 i2_prime $x186 i10_prime)) (not (and $x618 i4_prime $x297 i6_prime)) (not (and $x618 i4_prime $x714 i8_prime)) (not (and $x618 i4_prime $x186 i10_prime)) (not (and $x297 i6_prime $x714 i8_prime)) (not (and $x297 i6_prime $x186 i10_prime)) (not (and $x186 i10_prime $x714 i8_prime)))))
 (let (($x444 (not $x557)))
 (not $x444))))))))))))))))))))
(check-sat)
