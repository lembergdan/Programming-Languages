
# Haskell
אינטרפרטר $\bf ghci$ 

כדי להריץ את האינטרפטר, מספיק להריץ ב-Terminal את הפקודה
ghci.

```bash
% ghci
GHCi, version 8.10.7: https://www.haskell.org/ghc/  :? @for help@
Prelude>
```

ב-ghci אפשר להריץ פקודות של Haskell.
```Haskell
Prelude> x = 10
Prelude> y = 20
Prelude> x + y
30
Prelude> x = [1,2,3,4]
Prelude> y = [2,3]
Prelude> z = x ++ y 
Prelude> z
[1,2,3,4,2,3]
```
ע׳׳י פקודות
:t
ו-
:info
אפשר לקבל מידע על אובייקטים ומחלקות.

```Haskell
Prelude> x = 10 :: Int
Prelude> y = [1,2,3,4]
Prelude> :t x
x :: Int
Prelude> :t y
y :: Num a => [a]
Prelude> :i x
x :: Int 	        -- Defined at <interactive>:32:1
Prelude> :i y
y :: Num a => [a] 	-- Defined at <interactive>:33:1
Prelude> :i Eq
type Eq :: * -> Constraint
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
  -- Defined @in@ 'GHC.Classes'
  ...
```
אפשר גם להריץ קובצ עם קוד באינטרפרטר. צריך להעבור לתיקיה של הקובץ ע׳׳י פקודה

```Haskell
Prelude> :cd path_to_file
```
ואז להריצ פקודה

```Haskell
Prelude> :load file_name
```


#### <ins>דוגמא</ins>
נבנה את הקובץ 
Fac.hs
עם קוד הבא:

```Haskell
main = do
    print (fac 20)

fac :: Int -> Int
fac 0 = 1
fac n = n * fac (n - 1)
```

נריץ

```Haskell
Prelude> :@cd@ /Users/lembergdan/Desktop/
Prelude> :load Fac
[1 of 1] Compiling Main             --( Fac.hs, interpreted )
Ok, one module loaded.
*Main> 
```

עכשיו אפשר להריץ פונקציות מהקובץ.

```Haskell
*Main> fac 10
3628800
*Main> 
```

### תכנית Hello~World

```Haskell
main = do
    print "Hello World!" 
________________________
"Hello World"
```

### פלט סטנדרתי

```Haskell
putStr   :: String -> IO()  
putStrLn :: String -> IO()  -- adds a newline  
print    :: a -> IO()
```

### קלט סטנדרתי

```Haskell
getLine :: IO String  
```

#### <ins>דוגמא</ins>

```Haskell
main = do
    text <- getLine
    putStrLn text
____________________
Hello World
Hello World
```
	
### קלט-פלט לקובץ

```Haskell
openFile      :: FilePath -> IOMode -> IO Handle
hClose        :: Handle -> IO() 
```

כאשר

```Haskell
FilePath =  String -- path names in the file system
IOMode   =  ReadMode | WriteMode | AppendMode | ReadWriteMode
```

```Haskell
hGetContents :: Handle -> IO String
hGetLine     :: Handle -> IO String
hGetChar     :: Handle -> IO Char
hPrint       :: Handle -> a -> IO()
hPutChar     :: Handle -> Char -> IO()
hPutStr      :: Handle -> String -> IO() 
hPutStrLn    :: Handle -> String -> IO() 
```

#### <ins>דוגמא</ins>

```Haskell
main = do
    let input_file  = "/Users/lembergdan/Desktop/untitled folder/in.txt"
    let output_file = "/Users/lembergdan/Desktop/untitled folder/out.txt"

    f    <- openFile input_file  ReadMode
    g    <- openFile output_file WriteMode 
    text <- hGetContents f

    hPutStr g text
    hClose f
    hClose g
```

