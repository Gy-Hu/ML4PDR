; benchmark generated from python API
(set-info :status unknown)
(declare-fun v24 () Bool)
(declare-fun v14 () Bool)
(declare-fun v30 () Bool)
(declare-fun v28 () Bool)
(declare-fun v26 () Bool)
(declare-fun v22 () Bool)
(declare-fun v20 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12 () Bool)
(declare-fun i8 () Bool)
(assert
 (= v24 true))
(assert
 (= v14 true))
(assert
 (= v30 false))
(assert
 (let (($x70 (not v30)))
 (let (($x354 (and v24 v14 $x70)))
 (let (($x55 (not v28)))
 (let (($x50 (not v26)))
 (let (($x51 (not v24)))
 (let (($x45 (not v22)))
 (let (($x46 (not v20)))
 (let (($x40 (not v18)))
 (let (($x41 (not v16)))
 (let (($x35 (not v14)))
 (let (($x36 (not v12)))
 (let (($x141 (and $x36 $x35 $x41 $x40 $x46 $x45 $x51 $x50 $x55 $x70)))
 (let (($x554 (and $x141)))
 (let (($x429 (and $x554 (not $x354) (and (and (not (and $x50 $x51)) i8) v18 (not $x35)))))
 (not (and (not $x429) (not (and $x554 $x354)))))))))))))))))))
(check-sat)
