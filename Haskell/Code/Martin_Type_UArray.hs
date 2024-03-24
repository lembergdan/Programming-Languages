import Data.Array.ST
import Control.Monad 
import Data.Array.Unboxed

data Mat = Mat { arr  :: UArray Int Float, rows :: Int, cols :: Int }

instance Show Mat where 
    show :: Mat -> String
    show Mat {arr = arr_, rows = m, cols = n} = toStr (elems arr_) m
        where 
            toStr :: [Float] -> Int -> String 
            toStr l m | m == 1    = show l 
                      | otherwise = show (take n l) ++ "\n" ++ toStr (drop n l) (m - 1)

fromLists :: [[Float]] -> Mat
fromLists l = Mat {arr = listArray (0, m * n - 1) (concat l), rows = m, cols = n}
        where
            m = length l
            n = length $ head l

binOp :: (Float -> Float -> Float) -> Mat -> Mat -> Mat
binOp op mat1 mat2 = Mat { arr = arr_, rows = m, cols = n }
    where 
        m  = rows mat1
        n  = cols mat1 
        arr_ = binOp_ (arr mat1) (arr mat2)

        binOp_ a b = runSTUArray $ do 
            newArr <- newArray (0, m * n - 1) 0

            forM_ [0 .. (m - 1)] $ \i -> do 
                forM_ [0 .. (n - 1)] $ \j -> do  
                    let x = a ! (i * n + j) 
                    let y = b ! (i * n + j) 
                    writeArray newArr (i * n + j) (op x y)                   
            return newArr

uniOp :: (Float -> Float) -> Mat -> Mat
uniOp op mat = Mat { arr = arr_, rows = m, cols = n }
    where 
        m    = rows mat
        n    = cols mat 
        arr_ = uniOp_ (arr mat)

        uniOp_ a = runSTUArray $ do 
            newArr <- newArray (0, m * n - 1) 0

            forM_ [0 .. (m * n - 1)] $ \i -> do  
                    let x = a ! i 
                    writeArray newArr i (op x)                   
            return newArr

instance Num Mat where    
    mat1 + mat2 = binOp (+) mat1 mat2 
    mat1 - mat2 = binOp (-) mat1 mat2 
    mat1 * mat2 = binOp (*) mat1 mat2 
    signum mat  = uniOp signum mat 
    abs mat     = uniOp abs mat 
    fromInteger mat = Mat { arr = listArray (0,0) [fromInteger mat], rows = 1, cols = 1 } 

main = do 
    let l1 = [[1,2,3,4], [2,3,4,5], [3,4,5,6]] :: [[Float]]
        l2 = [[1,2,5,4], [2,6,4,7], [1,1,5,1]] :: [[Float]]

        a = fromLists l1
        b = fromLists l2

        c = a + b 
        d = a * b 

    print a; putStrLn ""
    print b; putStrLn ""
    print c; putStrLn ""
    print d; putStrLn ""
