; benchmark generated from python API
(set-info :status unknown)
(declare-fun v20 () Bool)
(declare-fun v26 () Bool)
(declare-fun v30 () Bool)
(declare-fun v28 () Bool)
(declare-fun v24 () Bool)
(declare-fun v22 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v14 () Bool)
(declare-fun v12 () Bool)
(declare-fun i6 () Bool)
(declare-fun i8 () Bool)
(declare-fun i10 () Bool)
(declare-fun i4 () Bool)
(declare-fun i2 () Bool)
(assert
 (= v20 true))
(assert
 (= v26 true))
(assert
 (= v30 false))
(assert
 (let (($x70 (not v30)))
 (let (($x350 (and v20 v26 $x70)))
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
 (let (($x245 (not (and v18 v26))))
 (let (($x329 (not (and v22 $x70))))
 (let (($x480 (not (and v24 $x70))))
 (let (($x68 (not (and v26 v24))))
 (let (($x74 (not i10)))
 (let (($x66 (not (and v22 v20))))
 (let (($x64 (not (and v14 v12))))
 (let (($x63 (not (and v18 v16))))
 (let (($x72 (not (and $x70 v28))))
 (let (($x254 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74)) $x68))))
 (let (($x244 (not (and $x72 (not (and $x63 $x64 $x66 $x68))))))
 (let (($x230 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 (not i8))) $x66))))
 (let (($x239 (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 (not i8) (not i6))) $x63)))
 (let (($x240 (not $x239)))
 (let (($x102 (not i4)))
 (let (($x93 (not i6)))
 (let (($x85 (not i8)))
 (let (($x266 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 $x85 $x93 $x102)) $x64))))
 (let (($x278 (and (not (and $x266 i2 $x240 i4)) (not (and $x266 i2 $x230 i6)) (not (and $x266 i2 $x254 i8)) (not (and $x266 i2 $x244 i10)) (not (and $x240 i4 $x230 i6)) (not (and $x240 i4 $x254 i8)) (not (and $x240 i4 $x244 i10)) (not (and $x230 i6 $x254 i8)) (not (and $x230 i6 $x244 i10)) (not (and $x244 i10 $x254 i8)))))
 (let (($x355 (and (and $x278 $x40 $x35 $x46 $x41 $x480 $x329 $x245 $x36) (not $x350) (and (and (not (and $x45 $x46)) i6) $x70 (not $x35)))))
 (not (and (not $x355) (not (and (and $x141) $x350)))))))))))))))))))))))))))))))))))))
(check-sat)
