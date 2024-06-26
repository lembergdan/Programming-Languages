בשאלות הבאות יש להשתמש בסוג ״עץ חיפוש בינארי״, המוגדר באופן הבא:
```Haskell
data Tree a = Nil | Tree (Tree a) a (Tree a)
```

___
### שאלה 1.

כתבו את הפונקציה הבאה:
#### קלט:
עץ 
$t$
ואיבר 
$x$
#### פלט:
עץ t בתוספת $x$.
___
### שאלה 2.

כתבו את הפונקציה הבאה:
#### קלט:
רשימה 
$l$
#### פלט:
עץ t המכיל כל איברי הרשימה $l$.
___
### שאלה 3.

כתבו את הפונקציה הבאה:
#### קלט:
עץ 
$t$
#### פלט:
רשימה $l$ המכילה את כל האיברי העץ $t$.

___
### שאלה 4.

כתבו את הפונקציה הבאה:
#### קלט:
עץ 
$t$
ומספר טבעי
$n$
#### פלט:
רשימה $l$ המכילה את כל האיברי ה- $level$ מספר $n$ בעץ $t$.
___
### שאלה 5.

כתבו את הפונקציה הבאה:
#### קלט:
עץ 
$t$
(לא בהכרח $BST$)
$n$
#### פלט:
ערך 
$True$
אם עץ $t$ הוא $BST$, אחרת $False$.
___
### שאלה 6.

כתבו את הפונקציה הבאה:
#### קלט:
עץ 
$t$
(לא בהכרח $BST$)
$n$
#### פלט:
תת-עץ 
$BST$
מקסימלי בעץ $t$.
___
### שאלה 7.

כתבו את הפונקציה הבאה:
#### קלט:
עץ של מספרים שלמים 
$t$
(לא בהכרח $BST$)
#### פלט:
תת-עץ הגדול ביותר, המכיל רק מספרים אי-שליליים.
___

### שאלה 8.

כתבו את הפונקציה הבאה:
#### קלט:
עץ של מספרים שלמים 
$t$
(לא בהכרח $BST$)
#### פלט:
ענף הקבד ביותר.
</br>

**כאשר:** 
</br>
**ענף** הוא רשימת הקדקדים 
$[a_1,a_2,\ldots,a_n]$
בה 
$a_{i+1}$
הוא בן של 
$a_i$
ו- 
$a_n$
הוא עלה.
</br>
**משקל של ענף** הוא סחום הקדקודים בו.
___


