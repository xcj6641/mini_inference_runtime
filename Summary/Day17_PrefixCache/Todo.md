1. When block manager releasing blocks, it doesn't evict key:block_id entry, so prefix_cache still contains the key:block_id entry even if the block is already free, which may cause error blocks problem.

Questions:
1. left padding v.s. right padding