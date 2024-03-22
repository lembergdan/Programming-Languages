___
### שאלה 1.

כתבו את הפונקציה הבאה:
#### קלט:
שני
`UArray Int Double`
מערכים 
$a, b$
#### פלט:
מכפלה פנימית של $a$ ו- $b$ֿ.
### הערה:
בשאלה זו כדי להשתמש בפונקציות הבאות:
```Haskell 
bounds :: (IArray a e, Ix i) => a i e -> (i, i)
```
```Haskell 
(!) :: (IArray a e, Ix i) => a i e -> i -> e
```
___
### שאלה 2.

כתבו את הפונקציה הבאה:
#### קלט:
שני
`UArray Int Double`
מערכים 
$a, b$
#### פלט:
מכפלה פנימית של $a$ ו- $b$ֿ.
### הערה:
בשאלה זו כדי להשתמש בפונקציות הבאות:
```Haskell 
bounds :: (IArray a e, Ix i) => a i e -> (i, i)
```
```Haskell 
(!) :: (IArray a e, Ix i) => a i e -> i -> e
```
___
### שאלה 3.

כתבו את הפונקציה הבאה:
#### קלט:
שני
`UArray Int Double`
מערכים 
$a, b$
#### פלט:
מערך 
`UArray Int Double`
המכיל את הסכום של $a$ ו- $b$ֿ.
### הערה:
בשאלה זו כדי להשתמש בפונקציות הבאות:

```Haskell 
runSTUArray :: (forall s. ST s (STUArray s i e)) -> UArray i e
```
```Haskell 
newArray :: (MArray a e m, Ix i) => (i, i) -> e -> m (a i e)
```
```Haskell 
writeArray :: (MArray a e m, Ix i) => a i e -> i -> e -> m ()
```
```Haskell 
bounds :: (IArray a e, Ix i) => a i e -> (i, i)
```
```Haskell 
(!) :: (IArray a e, Ix i) => a i e -> i -> e
```
```Haskell 
forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
```
___
### שאלה 4.

כתבו את הפונקציה הבאה:
#### קלט:
מטריצה $A$, כמערך
`UArray Int Double`
ווקטור 
$v$
כמערך
`UArray Int Double`
#### פלט:
מכפלה
$A \cdot v$
כמערך
`UArray Int Double`
### הערה:
בשאלה זו כדי להשתמש בפונקציות הבאות:

```Haskell 
runSTUArray :: (forall s. ST s (STUArray s i e)) -> UArray i e
```
```Haskell 
newArray :: (MArray a e m, Ix i) => (i, i) -> e -> m (a i e)
```
```Haskell 
writeArray :: (MArray a e m, Ix i) => a i e -> i -> e -> m ()
```
```Haskell 
bounds :: (IArray a e, Ix i) => a i e -> (i, i)
```
```Haskell 
(!) :: (IArray a e, Ix i) => a i e -> i -> e
```
```Haskell 
forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
```
```Haskell 
forM :: (Traversable t, Monad m) => t a -> (a -> m b) -> m (t b)
```
___
### שאלה 5.

כתבו את הפונקציה הבאה:
#### קלט:
שתי מטריצות 
$A, B$, 
כמערכים
`UArray Int Double`
#### פלט:
מכפלה
$A \cdot B$
כמערך
`UArray Int Double`
### הערה:
בשאלה זו כדי להשתמש בפונקציות הבאות:

```Haskell 
runSTUArray :: (forall s. ST s (STUArray s i e)) -> UArray i e
```
```Haskell 
newArray :: (MArray a e m, Ix i) => (i, i) -> e -> m (a i e)
```
```Haskell 
writeArray :: (MArray a e m, Ix i) => a i e -> i -> e -> m ()
```
```Haskell 
bounds :: (IArray a e, Ix i) => a i e -> (i, i)
```
```Haskell 
(!) :: (IArray a e, Ix i) => a i e -> i -> e
```
```Haskell 
forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
```
___


