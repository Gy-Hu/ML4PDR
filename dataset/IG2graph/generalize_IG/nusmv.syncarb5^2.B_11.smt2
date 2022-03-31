; benchmark generated from python API
(set-info :status unknown)
(declare-fun v20 () Bool)
(declare-fun v14 () Bool)
(declare-fun v28 () Bool)
(declare-fun v26 () Bool)
(declare-fun v30 () Bool)
(declare-fun v24 () Bool)
(declare-fun v22 () Bool)
(declare-fun v18 () Bool)
(declare-fun v16 () Bool)
(declare-fun v12 () Bool)
(declare-fun i10 () Bool)
(declare-fun i6 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v28_prime () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v20_prime () Bool)
(declare-fun v18_prime () Bool)
(declare-fun v16_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v14_prime () Bool)
(declare-fun v12_prime () Bool)
(declare-fun i2 () Bool)
(assert
 (= v20 true))
(assert
 (= v14 true))
(assert
 (= v28 true))
(assert
 (= v26 true))
(assert
 (let (($x201 (and v20 v14 v28 v26)))
 (let (($x70 (not v30)))
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
 (let (($x57 (not (and v30 $x55))))
 (let (($x58 (and $x57 i10)))
 (let (($x48 (not (and $x45 $x46))))
 (let (($x49 (and $x48 i6)))
 (let (($x398 (not (and $x35 (not v30_prime)))))
 (let (($x359 (not (and v30_prime v14))))
 (let (($x585 (not (and $x57 i10 (not v28_prime)))))
 (let (($x600 (not (and v28_prime (not $x58)))))
 (let (($x543 (not (and $x70 (not v26_prime)))))
 (let (($x636 (not (and v26_prime v30))))
 (let (($x319 (not (and (not (and $x50 $x51)) i8 (not v24_prime)))))
 (let (($x403 (not (and v24_prime (not (and (not (and $x50 $x51)) i8))))))
 (let (($x185 (not (and v26 (not v22_prime)))))
 (let (($x487 (not (and v22_prime $x50))))
 (let (($x423 (not (and $x48 i6 (not v20_prime)))))
 (let (($x479 (not (and v20_prime (not $x49)))))
 (let (($x488 (not (and v22 (not v18_prime)))))
 (let (($x597 (not (and v18_prime $x45))))
 (let (($x513 (not (and (not (and $x40 $x41)) i4 (not v16_prime)))))
 (let (($x637 (not (and v16_prime (not (and (not (and $x40 $x41)) i4))))))
 (let (($x497 (not (and v18 (not v14_prime)))))
 (let (($x512 (not (and v14_prime $x40))))
 (let (($x378 (not (and (not (and $x35 $x36)) i2 (not v12_prime)))))
 (let (($x379 (not (and v12_prime (not (and (not (and $x35 $x36)) i2))))))
 (let (($x437 (and $x379 $x378 $x512 $x497 $x637 $x513 $x597 $x488 $x479 $x423 $x487 $x185 $x403 $x319 $x636 $x543 $x600 $x585 $x359 $x398)))
 (let (($x579 (not $x201)))
 (let (($x298 (not (and v14 v22))))
 (let (($x303 (not (and v24 $x70))))
 (let (($x575 (not (and v20 $x70))))
 (let (($x515 (not (and v18 v26))))
 (let (($x495 (not (and v22 $x70))))
 (let (($x68 (not (and v26 v24))))
 (let (($x74 (not i10)))
 (let (($x66 (not (and v22 v20))))
 (let (($x64 (not (and v14 v12))))
 (let (($x63 (not (and v18 v16))))
 (let (($x72 (not (and $x70 v28))))
 (let (($x254 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74)) $x68))))
 (let (($x244 (not (and $x72 (not (and $x63 $x64 $x66 $x68))))))
 (let (($x277 (not (and $x244 i10 $x254 i8))))
 (let (($x230 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 (not i8))) $x66))))
 (let (($x275 (not (and $x230 i6 $x244 i10))))
 (let (($x273 (not (and $x230 i6 $x254 i8))))
 (let (($x239 (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 (not i8) (not i6))) $x63)))
 (let (($x240 (not $x239)))
 (let (($x271 (not (and $x240 i4 $x244 i10))))
 (let (($x269 (not (and $x240 i4 $x254 i8))))
 (let (($x237 (not (and $x240 i4 $x230 i6))))
 (let (($x102 (not i4)))
 (let (($x93 (not i6)))
 (let (($x85 (not i8)))
 (let (($x266 (not (and (not (and $x72 $x63 $x64 $x66 $x68 $x74 $x85 $x93 $x102)) $x64))))
 (let (($x243 (not (and $x266 i2 $x244 i10))))
 (let (($x253 (not (and $x266 i2 $x254 i8))))
 (let (($x263 (not (and $x266 i2 $x230 i6))))
 (let (($x265 (not (and $x266 i2 $x240 i4))))
 (let (($x278 (and $x265 $x263 $x253 $x243 $x237 $x269 $x271 $x273 $x275 $x277)))
 (let (($x558 (and (and $x278 $x35 $x41 $x495 $x515 $x36 $x575 $x303 $x298) $x579 $x437 (and $x49 v18 $x58 $x70))))
 (not (and (not $x558) (not (and (and $x141) $x201)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
