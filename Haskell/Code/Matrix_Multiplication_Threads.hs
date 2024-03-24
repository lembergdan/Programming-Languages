mult :: (Int, Int) -> Int -> Int -> UArray Int Double -> UArray Int Double -> UArray Int Double 
mult (m1, m2) k n a b = runSTUArray $ do 
    arr <- newArray (0, (m2 - m1 + 1) * n - 1) 0.0 
    forM_[m1 .. m2] $ \i -> do
        forM_[0 .. n - 1] $ \j -> do
            writeArray arr ((i - m1) * n + j) (dot i j 0 0)
    return arr 

    where
        {-# INLINE getA #-}
        getA i j = a ! (i * k + j)
        {-# INLINE getB #-}
        getB i j = b ! (i * n + j)

        dot :: Int -> Int -> Int -> Double -> Double
        dot i j t acc | t == k    = acc
                      | otherwise = dot i j (t + 1) (acc + (getA i t * getB t j))

                      
multMat :: Int -> Int -> Int -> UArray Int Double -> UArray Int Double -> UArray Int Double 
multMat m k n a b = listArray (0, m * n - 1) (c1 `par` c2 `par` c3 `par` c4 `seq` (c1 ++ c2 ++ c3 ++ c4))
    where 
        t = m `div` 4
        c1 = elems $ mult (0 * t, 1 * t - 1) k n a b
        c2 = elems $ mult (1 * t, 2 * t - 1) k n a b
        c3 = elems $ mult (2 * t, 3 * t - 1) k n a b
        c4 = elems $ mult (3 * t, 4 * t - 1) k n a b
