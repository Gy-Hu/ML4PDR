; benchmark generated from python API
(set-info :status unknown)
(declare-fun v16 () Bool)
(declare-fun v26 () Bool)
(declare-fun v30 () Bool)
(declare-fun v28 () Bool)
(declare-fun v24 () Bool)
(declare-fun v22 () Bool)
(declare-fun v20 () Bool)
(declare-fun v18 () Bool)
(declare-fun v14 () Bool)
(declare-fun v12 () Bool)
(declare-fun i4 () Bool)
(assert
 (= v16 true))
(assert
 (= v26 true))
(assert
 (= v30 false))
(assert
 (let (($x76 (not v30)))
 (let (($x581 (and v16 v26 $x76)))
 (let (($x70 (not v28)))
 (let (($x71 (not v26)))
 (let (($x65 (not v24)))
 (let (($x66 (not v22)))
 (let (($x504 (not v20)))
 (let (($x449 (not v18)))
 (let (($x492 (not v16)))
 (let (($x380 (not v14)))
 (let (($x639 (not v12)))
 (let (($x10 (and $x639 $x380 $x492 $x449 $x504 $x66 $x65 $x71 $x70 $x76)))
 (let (($x617 (and $x10)))
 (let (($x167 (and $x617 (not $x581) (and (and (not (and $x449 $x492)) i4) $x76 (not $x380)))))
 (not (and (not $x167) (not (and $x617 $x581)))))))))))))))))))
(check-sat)
