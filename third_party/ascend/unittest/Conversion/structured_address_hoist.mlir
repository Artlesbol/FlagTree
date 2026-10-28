// RUN: triton-mlir-opt %s --hoist-structured-permutation-addresses --allow-unregistered-dialect


func.func private @custom_flagtree_stride_gather_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @consume(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @mutate(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func @packed_double(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @packed_single(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %ss = memref.cast %sa : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @unaligned_source(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 388 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @index_extra_read(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      call @consume(%ii) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    
    return
}

func.func @physical_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %alias = hivm.hir.pointer_cast(%ca) : memref<32xi32, #hivm.address_space<ub>>
      %alias_d = memref.cast %alias : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @consume(%alias_d) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    
    return
}

func.func @unknown_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      call @consume(%unknown) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    
    return
}

func.func @index_write(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      call @mutate(%ii) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    
    return
}

func.func @after_loop_read(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    call @consume(%ii) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    return
}

func.func @strided_indices(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, strided<[2]>, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, strided<[2]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @unaligned_indices(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 4 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @unaligned_scratch(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 132 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @unaligned_destination(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 260 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @overlapping_tables(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 0 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @source_destination_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 256 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @conditional_gather(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      scf.if %cond {
        call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      }
      
    }
    
    return
}

func.func @unknown_source(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%dynamic) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}

func.func @outside_loop_read_before(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @consume(%ii) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      
    }
    
    return
}
