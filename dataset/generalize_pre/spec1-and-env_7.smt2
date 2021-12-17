; benchmark generated from python API
(set-info :status unknown)
(declare-fun v52 () Bool)
(declare-fun v28 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v12 () Bool)
(declare-fun v14 () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v20 () Bool)
(declare-fun v48 () Bool)
(declare-fun i6 () Bool)
(declare-fun v30 () Bool)
(declare-fun v16_prime () Bool)
(declare-fun v32 () Bool)
(declare-fun v42 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun i10 () Bool)
(declare-fun v44_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v48_prime () Bool)
(declare-fun v12_prime () Bool)
(declare-fun v50_prime () Bool)
(declare-fun v44 () Bool)
(declare-fun i2 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v54 () Bool)
(declare-fun v54_prime () Bool)
(declare-fun v16 () Bool)
(declare-fun v34 () Bool)
(declare-fun v24 () Bool)
(declare-fun v32_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v42_prime () Bool)
(declare-fun v18 () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v46 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v38_prime () Bool)
(declare-fun v36 () Bool)
(declare-fun v34_prime () Bool)
(declare-fun v22 () Bool)
(declare-fun v36_prime () Bool)
(declare-fun v50 () Bool)
(declare-fun v40_prime () Bool)
(declare-fun v46_prime () Bool)
(declare-fun v52_prime () Bool)
(declare-fun v40 () Bool)
(declare-fun v38 () Bool)
(assert
 (= v52 true))
(assert
 (= v28 true))
(assert
 (= v28_prime true))
(assert
 (= v12 false))
(assert
 (= v14 false))
(assert
 (= v20_prime false))
(assert
 (= v20 false))
(assert
 (= v48 false))
(assert
 (= i6 false))
(assert
 (= v30 false))
(assert
 (= v16_prime true))
(assert
 (= v32 false))
(assert
 (= v42 false))
(assert
 (= v22_prime false))
(assert
 (= v24_prime true))
(assert
 (= i10 false))
(assert
 (= v44_prime false))
(assert
 (= i4 false))
(assert
 (= v48_prime false))
(assert
 (= v12_prime true))
(assert
 (= v50_prime false))
(assert
 (= v44 false))
(assert
 (= i2 false))
(assert
 (= v26_prime false))
(assert
 (= v54 true))
(assert
 (= v54_prime true))
(assert
 (= v16 true))
(assert
 (= v34 true))
(assert
 (= v24 false))
(assert
 (= v32_prime false))
(assert
 (= v26 true))
(assert
 (= v30_prime false))
(assert
 (= v42_prime false))
(assert
 (= v18 true))
(assert
 (= v18_prime false))
(assert
 (= v46 true))
(assert
 (= v14_prime false))
(assert
 (= i8 false))
(assert
 (= v38_prime false))
(assert
 (= v36 true))
(assert
 (= v34_prime false))
(assert
 (= v22 true))
(assert
 (= v36_prime false))
(assert
 (= v50 true))
(assert
 (= v40_prime true))
(assert
 (= v46_prime true))
(assert
 (= v52_prime false))
(assert
 (= v40 true))
(assert
 (= v38 false))
(assert
 (let (($x109 (not v12)))
 (let (($x72 (not v28)))
 (let (($x67 (and (not (and (not v18) v16)) (not (and (and v54 v16) v18)))))
 (let (($x80 (and (not (and (and (not (and (and v54 v16) v20)) v22) v28)) (not (and (and (not (and (not $x67) v20)) v22) $x72)))))
 (let (($x104 (not v36)))
 (let (($x90 (not v20)))
 (let (($x89 (not (and (not (and (not v52) v16)) (not (and (not $x67) v52))))))
 (let (($x101 (not (and (not (and (not (and (and v54 v16) $x90)) v22)) v28))))
 (let (($x105 (and (and $x101 (not (and (not (and (not (and $x89 $x90)) v22)) $x72))) $x104)))
 (let (($x108 (and (not $x105) (not (and (not $x80) v36)))))
 (let (($x115 (and (not (and (not (and $x108 $x109)) v40)) (not (and (not v40) v12)))))
 (let (($x1995 (not $x115)))
 (not (not (not $x1995))))))))))))))))
(check-sat)
