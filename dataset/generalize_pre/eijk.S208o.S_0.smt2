; benchmark generated from python API
(set-info :status unknown)
(declare-fun i8_prime () Bool)
(declare-fun v36_prime () Bool)
(declare-fun v32 () Bool)
(declare-fun i12 () Bool)
(declare-fun i16_prime () Bool)
(declare-fun i10_prime () Bool)
(declare-fun i20 () Bool)
(declare-fun v46 () Bool)
(declare-fun v44 () Bool)
(declare-fun v50_prime () Bool)
(declare-fun v42 () Bool)
(declare-fun i20_prime () Bool)
(declare-fun v48 () Bool)
(declare-fun i18_prime () Bool)
(declare-fun v52_prime () Bool)
(declare-fun v50 () Bool)
(declare-fun i14 () Bool)
(declare-fun i2 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i8 () Bool)
(declare-fun v26 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v42_prime () Bool)
(declare-fun i4_prime () Bool)
(declare-fun v36 () Bool)
(declare-fun i16 () Bool)
(declare-fun i6_prime () Bool)
(declare-fun v34_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun v34 () Bool)
(declare-fun v38 () Bool)
(declare-fun v40_prime () Bool)
(declare-fun v22 () Bool)
(declare-fun v28 () Bool)
(declare-fun v32_prime () Bool)
(declare-fun i12_prime () Bool)
(declare-fun v52 () Bool)
(declare-fun v46_prime () Bool)
(declare-fun v44_prime () Bool)
(declare-fun v38_prime () Bool)
(declare-fun i2_prime () Bool)
(declare-fun i4 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun i18 () Bool)
(declare-fun i14_prime () Bool)
(declare-fun i6 () Bool)
(declare-fun v40 () Bool)
(declare-fun i10 () Bool)
(declare-fun v48_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v30_prime () Bool)
(assert
 (= i8_prime false))
(assert
 (= v36_prime false))
(assert
 (= v32 true))
(assert
 (= i12 false))
(assert
 (= i16_prime false))
(assert
 (= i10_prime true))
(assert
 (= i20 false))
(assert
 (= v46 true))
(assert
 (= v44 true))
(assert
 (= v50_prime false))
(assert
 (= v42 true))
(assert
 (= i20_prime false))
(assert
 (= v48 true))
(assert
 (= i18_prime false))
(assert
 (= v52_prime false))
(assert
 (= v50 true))
(assert
 (= i14 false))
(assert
 (= i2 true))
(assert
 (= v28_prime false))
(assert
 (= i8 false))
(assert
 (= v26 false))
(assert
 (= v26_prime false))
(assert
 (= v30 true))
(assert
 (= v42_prime false))
(assert
 (= i4_prime false))
(assert
 (= v36 true))
(assert
 (= i16 false))
(assert
 (= i6_prime false))
(assert
 (= v34_prime false))
(assert
 (= v24 false))
(assert
 (= v34 true))
(assert
 (= v38 true))
(assert
 (= v40_prime false))
(assert
 (= v22 false))
(assert
 (= v28 false))
(assert
 (= v32_prime false))
(assert
 (= i12_prime false))
(assert
 (= v52 true))
(assert
 (= v46_prime false))
(assert
 (= v44_prime false))
(assert
 (= v38_prime false))
(assert
 (= i2_prime true))
(assert
 (= i4 false))
(assert
 (= v22_prime true))
(assert
 (= i18 true))
(assert
 (= i14_prime false))
(assert
 (= i6 false))
(assert
 (= v40 true))
(assert
 (= i10 false))
(assert
 (= v48_prime false))
(assert
 (= v24_prime false))
(assert
 (= v30_prime false))
(assert
 (let (($x123 (and (not (and (and (and (and v32 v30) i2) v34) (not v36))) (not (and (not (and (and (and v32 v30) v34) i2)) v36)))))
 (let (($x297 (not $x123)))
 (let (($x114 (and (not (and (and (and v32 v30) i2) (not v34))) (not (and (not (and (and v32 v30) i2)) v34)))))
 (let (($x295 (not $x114)))
 (let (($x683 (not $x295)))
 (let (($x105 (not (and (not (and v30 i2)) (not v32)))))
 (let (($x106 (and (not (and (and v30 i2) v32)) $x105)))
 (let (($x640 (not $x106)))
 (let (($x92 (not v30)))
 (let (($x93 (and $x92 i2)))
 (let (($x97 (not $x93)))
 (let (($x96 (not (and v30 (not i2)))))
 (let (($x98 (and $x96 $x97)))
 (let (($x292 (not $x98)))
 (let (($x680 (not $x292)))
 (let (($x679 (not (and $x680 i2_prime $x640 $x683 $x297 i12_prime))))
 (let (($x659 (not $x297)))
 (let (($x61 (not v22)))
 (let (($x60 (and (and (and (and v32 v30) v34) v36) i2)))
 (let (($x285 (not (and (not (and (not $x60) v22)) (not (and $x60 $x61))))))
 (let (($x678 (not (and $x285 i2_prime $x640 $x680 $x659 $x683 i10_prime))))
 (let (($x68 (and $x60 v22)))
 (let (($x72 (and $x68 v24)))
 (let (($x73 (not $x72)))
 (let (($x75 (and $x73 (not (and (not $x68) (not v24))))))
 (let (($x682 (not $x285)))
 (let (($x688 (not (and $x682 i2_prime $x75 $x640 $x680 $x659 $x683 i8_prime))))
 (let (($x288 (not (and (not (and $x73 v26)) (not (and $x72 (not v26)))))))
 (let (($x690 (not $x75)))
 (let (($x681 (not (and $x682 i2_prime $x690 $x288 $x640 $x680 $x659 $x683 i6_prime))))
 (let (($x396 (not (and i20_prime i2_prime))))
 (let (($x689 (not (and $x292 i2_prime i18_prime))))
 (let (($x668 (not (and $x680 i2_prime $x106 i16_prime))))
 (let (($x671 (not (and $x680 i2_prime $x640 $x295 i14_prime))))
 (let (($x91 (and (not (and (and (not v28) v26) $x72)) (not (and (not (and (and v26 v24) $x68)) v28)))))
 (let (($x290 (not $x91)))
 (let (($x674 (and $x682 i2_prime $x690 (not $x288) $x290 $x640 $x680 $x659 $x683 i4_prime)))
 (let (($x667 (not $x674)))
 (let (($x189 (and (not (and (and (and (and v48 v46) i2) v50) (not v52))) (not (and (not (and (and (and v48 v46) v50) i2)) v52)))))
 (let (($x311 (not $x189)))
 (let (($x180 (and (not (and (and (and v48 v46) i2) (not v50))) (not (and (not (and (and v48 v46) i2)) v50)))))
 (let (($x309 (not $x180)))
 (let (($x663 (not $x309)))
 (let (($x171 (not (and (not (and v46 i2)) (not v48)))))
 (let (($x172 (and (not (and (and v46 i2) v48)) $x171)))
 (let (($x651 (not $x172)))
 (let (($x159 (not v46)))
 (let (($x160 (and $x159 i2)))
 (let (($x163 (not $x160)))
 (let (($x162 (not (and v46 (not i2)))))
 (let (($x164 (and $x162 $x163)))
 (let (($x306 (not $x164)))
 (let (($x666 (not $x306)))
 (let (($x655 (not (and $x666 i2_prime $x651 $x663 $x311 i12_prime))))
 (let (($x654 (not $x311)))
 (let (($x128 (not v38)))
 (let (($x127 (and (and (and (and v48 v46) v50) v52) i2)))
 (let (($x299 (not (and (not (and (not $x127) v38)) (not (and $x127 $x128))))))
 (let (($x638 (not (and $x299 i2_prime $x651 $x666 $x654 $x663 i10_prime))))
 (let (($x135 (and $x127 v38)))
 (let (($x139 (and $x135 v40)))
 (let (($x140 (not $x139)))
 (let (($x142 (and $x140 (not (and (not $x135) (not v40))))))
 (let (($x657 (not $x299)))
 (let (($x605 (not (and $x657 i2_prime $x142 $x651 $x666 $x654 $x663 i8_prime))))
 (let (($x302 (not (and (not (and $x140 v42)) (not (and $x139 (not v42)))))))
 (let (($x606 (not $x142)))
 (let (($x607 (not (and $x657 i2_prime $x606 $x302 $x651 $x666 $x654 $x663 i6_prime))))
 (let (($x641 (not (and $x306 i2_prime i18_prime))))
 (let (($x610 (not (and $x666 i2_prime $x172 i16_prime))))
 (let (($x613 (not (and $x666 i2_prime $x651 $x309 i14_prime))))
 (let (($x158 (and (not (and (and (not v44) v42) $x139)) (not (and (not (and (and v42 v40) $x135)) v44)))))
 (let (($x304 (not $x158)))
 (let (($x643 (and $x657 i2_prime $x606 (not $x302) $x304 $x651 $x666 $x654 $x663 i4_prime)))
 (let (($x615 (not $x643)))
 (let (($x617 (and (not (and $x615 $x613 $x610 $x641 $x396 $x607 $x605 $x638 $x655)) $x667 $x671 $x668 $x689 $x396 $x681 $x688 $x678 $x679)))
 (let (($x620 (and $x615 $x613 $x610 $x641 $x396 $x607 $x605 $x638 $x655 (not (and $x667 $x671 $x668 $x689 $x396 $x681 $x688 $x678 $x679)))))
 (let (($x621 (not (and (not $x620) (not $x617)))))
 (not $x621))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)
