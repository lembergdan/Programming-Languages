mult :: Int -> Int -> Int -> UArray Int Double -> UArray Int Double -> UArray Int Double 
mult m k n a b = runSTUArray $ do 
    arr <- newArray (0, m * n - 1) 0
    forM_[0 .. m - 1] $ \i -> do
        forM_[0 .. n - 1] $ \j -> do
            writeArray arr (i * n + j) (dot i j 0 0)
    return arr 

    where
        {-# INLINE getA #-}
        getA i j = a ! (i * k + j)
        {-# INLINE getB #-}
        getB i j = b ! (i * n + j)

        dot :: Int -> Int -> Int -> Double -> Double
        dot i j t acc | t == k    = acc
                      | otherwise = dot i j (t + 1) (acc + (getA i t * getB t j))
