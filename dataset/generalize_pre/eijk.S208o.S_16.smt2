; benchmark generated from python API
(set-info :status unknown)
(declare-fun i20 () Bool)
(declare-fun v52 () Bool)
(declare-fun i14_prime () Bool)
(declare-fun v28 () Bool)
(declare-fun v28_prime () Bool)
(declare-fun i20_prime () Bool)
(declare-fun v48 () Bool)
(declare-fun v42 () Bool)
(declare-fun v48_prime () Bool)
(declare-fun v30 () Bool)
(declare-fun v32 () Bool)
(declare-fun v22_prime () Bool)
(declare-fun v24_prime () Bool)
(declare-fun v50_prime () Bool)
(declare-fun i2 () Bool)
(declare-fun i18 () Bool)
(declare-fun v44_prime () Bool)
(declare-fun v44 () Bool)
(declare-fun v26_prime () Bool)
(declare-fun i16 () Bool)
(declare-fun v34 () Bool)
(declare-fun v32_prime () Bool)
(declare-fun v24 () Bool)
(declare-fun i2_prime () Bool)
(declare-fun v26 () Bool)
(declare-fun v30_prime () Bool)
(declare-fun v42_prime () Bool)
(declare-fun i16_prime () Bool)
(declare-fun v46 () Bool)
(declare-fun v38_prime () Bool)
(declare-fun v40 () Bool)
(declare-fun v52_prime () Bool)
(declare-fun v36 () Bool)
(declare-fun v34_prime () Bool)
(declare-fun v22 () Bool)
(declare-fun v46_prime () Bool)
(declare-fun i18_prime () Bool)
(declare-fun v50 () Bool)
(declare-fun v36_prime () Bool)
(declare-fun v40_prime () Bool)
(declare-fun v38 () Bool)
(declare-fun i12_prime () Bool)
(declare-fun i10_prime () Bool)
(declare-fun i8_prime () Bool)
(declare-fun i6_prime () Bool)
(declare-fun i4_prime () Bool)
(assert
 (= i20 false))
(assert
 (= v52 false))
(assert
 (= i14_prime false))
(assert
 (= v28 false))
(assert
 (= v28_prime false))
(assert
 (= i20_prime false))
(assert
 (= v48 true))
(assert
 (= v42 false))
(assert
 (= v48_prime false))
(assert
 (= v30 true))
(assert
 (= v32 false))
(assert
 (= v22_prime false))
(assert
 (= v24_prime false))
(assert
 (= v50_prime true))
(assert
 (= i2 true))
(assert
 (= i18 true))
(assert
 (= v44_prime false))
(assert
 (= v44 false))
(assert
 (= v26_prime false))
(assert
 (= i16 false))
(assert
 (= v34 false))
(assert
 (= v32_prime true))
(assert
 (= v24 false))
(assert
 (= i2_prime true))
(assert
 (= v26 false))
(assert
 (= v30_prime false))
(assert
 (= v42_prime false))
(assert
 (= i16_prime true))
(assert
 (= v46 true))
(assert
 (= v38_prime false))
(assert
 (= v40 false))
(assert
 (= v52_prime false))
(assert
 (= v36 false))
(assert
 (= v34_prime false))
(assert
 (= v22 false))
(assert
 (= v46_prime false))
(assert
 (= i18_prime false))
(assert
 (= v50 false))
(assert
 (= v36_prime false))
(assert
 (= v40_prime false))
(assert
 (= v38 false))
(assert
 (let (($x123 (and (not (and (and (and (and v32 v30) i2) v34) (not v36))) (not (and (not (and (and (and v32 v30) v34) i2)) v36)))))
 (let (($x297 (not $x123)))
 (let (($x114 (and (not (and (and (and v32 v30) i2) (not v34))) (not (and (not (and (and v32 v30) i2)) v34)))))
 (let (($x295 (not $x114)))
 (let (($x814 (not $x295)))
 (let (($x106 (and (not (and (and v30 i2) v32)) (not (and (not (and v30 i2)) (not v32))))))
 (let (($x420 (not $x106)))
 (let (($x292 (not (and (not (and v30 (not i2))) (not (and (not v30) i2))))))
 (let (($x696 (not $x292)))
 (let (($x663 (not $x297)))
 (let (($x61 (not v22)))
 (let (($x60 (and (and (and (and v32 v30) v34) v36) i2)))
 (let (($x285 (not (and (not (and (not $x60) v22)) (not (and $x60 $x61))))))
 (let (($x75 (and (not (and (and $x60 v22) v24)) (not (and (not (and $x60 v22)) (not v24))))))
 (let (($x1066 (not $x285)))
 (let (($x81 (and (not (and (not (and (and $x60 v22) v24)) v26)) (not (and (and (and $x60 v22) v24) (not v26))))))
 (let (($x288 (not $x81)))
 (let (($x440 (not (and $x1066 i2_prime (not $x75) $x288 $x420 $x696 $x663 $x814 i6_prime))))
 (let (($x1047 (not (and i20_prime i2_prime))))
 (let (($x91 (and (not (and (and (not v28) v26) (and (and $x60 v22) v24))) (not (and (not (and (and v26 v24) (and $x60 v22))) v28)))))
 (let (($x290 (not $x91)))
 (let (($x479 (and $x1066 i2_prime (not $x75) (not $x288) $x290 $x420 $x696 $x663 $x814 i4_prime)))
 (let (($x189 (and (not (and (and (and (and v48 v46) i2) v50) (not v52))) (not (and (not (and (and (and v48 v46) v50) i2)) v52)))))
 (let (($x311 (not $x189)))
 (let (($x180 (and (not (and (and (and v48 v46) i2) (not v50))) (not (and (not (and (and v48 v46) i2)) v50)))))
 (let (($x309 (not $x180)))
 (let (($x1061 (not $x309)))
 (let (($x172 (and (not (and (and v46 i2) v48)) (not (and (not (and v46 i2)) (not v48))))))
 (let (($x642 (not $x172)))
 (let (($x306 (not (and (not (and v46 (not i2))) (not (and (not v46) i2))))))
 (let (($x854 (not $x306)))
 (let (($x763 (not $x311)))
 (let (($x128 (not v38)))
 (let (($x127 (and (and (and (and v48 v46) v50) v52) i2)))
 (let (($x299 (not (and (not (and (not $x127) v38)) (not (and $x127 $x128))))))
 (let (($x142 (and (not (and (and $x127 v38) v40)) (not (and (not (and $x127 v38)) (not v40))))))
 (let (($x990 (not $x299)))
 (let (($x148 (and (not (and (not (and (and $x127 v38) v40)) v42)) (not (and (and (and $x127 v38) v40) (not v42))))))
 (let (($x302 (not $x148)))
 (let (($x756 (not (and $x990 i2_prime (not $x142) $x302 $x642 $x854 $x763 $x1061 i6_prime))))
 (let (($x158 (and (not (and (and (not v44) v42) (and (and $x127 v38) v40))) (not (and (not (and (and v42 v40) (and $x127 v38))) v44)))))
 (let (($x304 (not $x158)))
 (let (($x352 (and $x990 i2_prime (not $x142) (not $x302) $x304 $x642 $x854 $x763 $x1061 i4_prime)))
 (let (($x955 (and (not $x352) (not (and $x854 i2_prime $x642 $x309 i14_prime)) (not (and $x854 i2_prime $x172 i16_prime)) (not (and $x306 i2_prime i18_prime)) $x1047 $x756 (not (and $x990 i2_prime $x142 $x642 $x854 $x763 $x1061 i8_prime)) (not (and $x299 i2_prime $x642 $x854 $x763 $x1061 i10_prime)) (not (and $x854 i2_prime $x642 $x1061 $x311 i12_prime)))))
 (let (($x846 (and (not $x955) (not $x479) (not (and $x696 i2_prime $x420 $x295 i14_prime)) (not (and $x696 i2_prime $x106 i16_prime)) (not (and $x292 i2_prime i18_prime)) $x1047 $x440 (not (and $x1066 i2_prime $x75 $x420 $x696 $x663 $x814 i8_prime)) (not (and $x285 i2_prime $x420 $x696 $x663 $x814 i10_prime)) (not (and $x696 i2_prime $x420 $x814 $x297 i12_prime)))))
 (let (($x346 (and (not $x479) (not (and $x696 i2_prime $x420 $x295 i14_prime)) (not (and $x696 i2_prime $x106 i16_prime)) (not (and $x292 i2_prime i18_prime)) $x1047 $x440 (not (and $x1066 i2_prime $x75 $x420 $x696 $x663 $x814 i8_prime)) (not (and $x285 i2_prime $x420 $x696 $x663 $x814 i10_prime)) (not (and $x696 i2_prime $x420 $x814 $x297 i12_prime)))))
 (let (($x505 (and (not $x352) (not (and $x854 i2_prime $x642 $x309 i14_prime)) (not (and $x854 i2_prime $x172 i16_prime)) (not (and $x306 i2_prime i18_prime)) $x1047 $x756 (not (and $x990 i2_prime $x142 $x642 $x854 $x763 $x1061 i8_prime)) (not (and $x299 i2_prime $x642 $x854 $x763 $x1061 i10_prime)) (not (and $x854 i2_prime $x642 $x1061 $x311 i12_prime)) (not $x346))))
 (not (not (and (not $x505) (not $x846))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)