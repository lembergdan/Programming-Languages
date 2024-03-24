import Data.Array.ST 
import Control.Monad 
import Data.Array.Unboxed
import Data.Time.Clock
import Numeric.LinearAlgebra hiding ((!))

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

n = 1000

main = do 
    arr1 <- rand n n 
    arr2 <- rand n n 

    let !l1 = concat $ toLists arr1 
    let !l2 = concat $ toLists arr2

    let !a = listArray (0, n * n - 1) l1
    let !b = listArray (0, n * n - 1) l2

    tt1 <- getCurrentTime 
    let mm = mult n n n a b 
    print $ mm ! 2     
    tt2 <- getCurrentTime
    print (diffUTCTime tt2 tt1)
